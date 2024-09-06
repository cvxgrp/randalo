import functools
from typing import Callable, Literal
from dataclasses import dataclass, field

import numpy as np
import cvxpy as cp
import linops as lo
import torch

from . import modeling_layer as ml
from . import utils


def regularizer_sum_to_cvxpy(obj, variable):
    match obj:
        case ml.Sum(terms):
            return cp.sum([regularizer_sum_to_cvxpy(term, variable) for term in terms])
        case ml.SquareRegularizer(_):
            func = cp.sum_squares
        case ml.L1Regularizer(_):
            func = cp.norm1
        case ml.L2Regularizer(_):
            func = cp.norm2
        case ml.HuberRegularizer(_):
            func = cp.huber
        case _:
            raise RuntimeError("Unknown loss")
    expr = func(obj.linear @ variable if obj.linear is not None else variable)
    if obj.parameter is None:
        return obj.scale * expr
    else:
        return obj.scale * obj.parameter.scale * obj.parameter.parameter * expr


def loss_to_cvxpy(obj, X, y, variable):
    match obj:
        case ml.LogisticLoss():
            return cp.sum(cp.logistic(-cp.multiply(y, X @ variable)))
        case ml.MSELoss():
            return cp.sum_squares(y - X @ variable) / np.prod(y.shape) / 2
        case _:
            raise RuntimeError("Unknown loss")


def transform_model_to_cvxpy(loss, regularizer, X, y, variable):
    return cp.Problem(
        cp.Minimize(
            loss_to_cvxpy(loss, X, y, variable)
            + regularizer_sum_to_cvxpy(regularizer, variable)
        )
    )


class Jacobian(lo.LinearOperator):
    solution_func: Callable[[], torch.Tensor]
    loss: ml.Loss
    regularizer: ml.Sum | ml.Regularizer
    inverse_method: Literal[None, "minres", "cholesky"]

    supports_operator_matrix = True

    def __init__(self, y, X, solution_func, loss, regularizer, inverse_method=None):
        self.solution_func = solution_func
        self.loss = loss
        self.regularizer = regularizer
        self.inverse_method = inverse_method
        self.y = utils.to_tensor(y)
        self.X = utils.to_tensor(X)

    @property
    def _shape(self):
        n = self.y.shape[0]
        return (n, n)

    _diag: torch.Tensor = None

    @functools.cached_property
    def diag(self):
        return torch.diag(self @ torch.eye(self.shape[1]))

    def _matmul_impl(self, rhs):
        rhs = utils.to_tensor(rhs)
        needs_squeeze = False
        if len(rhs.shape) == 1:
            rhs = rhs.unsqueeze(-1)
            needs_squeeze = True
        beta_hat = utils.to_tensor(self.solution_func())
        y = self.y
        X = self.X
        _, _, _, d2loss_dboth, d2loss_dy_hat2 = utils.compute_derivatives(
            self.loss, y, X @ beta_hat
        )

        mask = torch.ones_like(beta_hat.squeeze(), dtype=bool)

        constraints, hessians = unpack_regularizer(self.regularizer, mask, beta_hat)
        X_mask = X[:, mask]
        rhs_scaled = -d2loss_dboth[:, None] * rhs

        if constraints is None and hessians is None:
            sqrt_d2loss_dy_hat2 = torch.sqrt(d2loss_dy_hat2)[:, None]
            tilde_X = sqrt_d2loss_dy_hat2 * X_mask
            Q, _ = torch.linalg.qr(tilde_X)
            return (
                Q @ (Q.T @ (rhs_scaled / sqrt_d2loss_dy_hat2))
            ) / sqrt_d2loss_dy_hat2
        elif constraints is None:
            kkt_rhs = X_mask.T @ rhs_scaled
            if hessians is None:
                tilde_X = torch.sqrt(d2loss_dy_hat2)[:, None] * X_mask
                _, R = torch.linalg.qr(tilde_X, mode="r")
            else:
                hessians_mask = hessians[mask, :][:, mask]
                P = X_mask.T @ (d2loss_dy_hat2[:, None] * X_mask) + hessians_mask
                R = torch.linalg.cholesky(P, upper=True)
            v = torch.linalg.solve_triangular(
                R, torch.linalg.solve_triangular(R.T, kkt_rhs, upper=False), upper=True
            )
        else:
            constraints_mask = constraints[:, mask]
            n, m = constraints_mask.shape
            if n >= m:
                _, N = torch.linalg.qr(constraints_mask, mode="r")
            else:
                N = constraints_mask

            if hessians is None:
                tilde_X = torch.sqrt(d2loss_dy_hat2)[:, None] * X_mask
                _, P_R = torch.linalg.qr(tilde_X, mode="r")
            else:
                hessians_mask = hessians[mask, :][:, mask]
                P = X_mask.T @ (d2loss_dy_hat2[:, None] * X_mask) + hessians_mask
                P_R = torch.linalg.cholesky(P, upper=True)

            S = self.D_nmask @ torch.linalg.solve_triangular(
                P_R,
                torch.linalg.solve_triangular(P_R.T, kkt_rhs, upper=False),
                upper=True,
            )
            S_R = torch.linalg.cholesky(S, upper=True)
            NPinvRhs = N @ torch.linalg.solve_triangular(
                P_R,
                torch.linalg.solve_triangular(P_R.T, kkt_rhs, upper=False),
                upper=True,
            )
            nu = torch.linalg.solve_triangular(
                S_R,
                torch.linalg.solve_triangular(S_R.T, -NPinvRhs, upper=False),
                upper=True,
            )
            v = torch.linalg.solve_triangular(
                P_R,
                torch.linalg.solve_triangular(P_R.T, kkt_rhs + N.T @ nu, upper=False),
                upper=True,
            )
        out = X_mask @ v
        return out if not needs_squeeze else out.squeeze(-1)


def unpack_regularizer(regularizer, mask, beta_hat, epsilon=1e-6):
    """
    Modifies mask in place

    Returns a constraint matrix (or None) and a hessian term (or None)
    """
    # Refactor this to have the ml object know how to form its constraint/hessian
    if not isinstance(regularizer, ml.Sum):
        if regularizer.parameter is not None:
            scale = regularizer.scale * regularizer.parameter.value
        else:
            scale = regularizer.scale

    match regularizer:
        case ml.Sum(exprs):
            constraints = []
            hessians = []
            for reg in exprs:
                cons, hess = unpack_regularizer(reg, mask, beta_hat)
                if cons is not None:
                    constraints.append(cons)
                if hess is not None:
                    hessians.append(hess)
            constraints = torch.vstack(constraints) if len(constraints) > 0 else None
            hessians = sum(hessians) if len(hessians) > 0 else None
            return constraints, hessians
        case ml.SquareRegularizer(linear):
            if linear is None:
                return None, torch.diag(
                    2 * scale * torch.ones_like(mask, dtype=beta_hat.dtype)
                )
            elif isinstance(linear, list):
                diag = torch.zeros_like(mask, dtype=beta_hat.dtype)
                diag[linear] = scale
                return None, torch.diag(diag)
            else:
                A = utils.to_tensor(linear)
                return None, torch.diag(scale * (A.mT @ A))
        case ml.L1Regularizer(linear):
            if linear is None:
                mask[torch.abs(beta_hat) <= epsilon] = False
                return None, None
            elif isinstance(linear, list):
                mask[linear][torch.abs(beta_hat[linear]) <= epsilon] = False
                return None, None
            else:
                A = utils.from_numpy(linear)
                return A[torch.abs(A @ beta_hat) <= epsilon, :], None

        case ml.L2Regularizer(linear):
            if linear is None:
                if torch.linalg.norm(beta_hat) <= epsilon:
                    mask[:] = False
                    return None, None
                else:
                    raise NotImplementedError("Hasn't been implemented yet")
                    return None, ...

            elif isinstance(linear, list):
                if torch.linalg.norm(beta_hat[linear]) <= epsilon:
                    mask[linear] = False
                    return None, None
                else:
                    raise NotImplementedError("Hasn't been implemented yet")
                    return None, ...
            else:
                if torch.linalg.norm(linear @ beta_hat) <= epsilon:
                    return linear, None
                else:
                    raise NotImplementedError("Hasn't been implemented yet")
                    return None, ...
        case ml.HuberRegularizer(linear, scale, parameter):
            raise NotImplementedError("TBD")


def transform_model_to_Jacobian(solution_func, loss, regularizer):
    return Jacobian(solution_func, loss, regularizer)
