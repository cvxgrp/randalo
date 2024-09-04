import functools
from typing import Callable, Literal
from dataclasses import dataclass, field

import cvxpy as cp
import linops as lo
import torch

from . import modeling_layer as ml


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

    if obj.parameter is None:
        return obj.scale * func(obj.linear @ variable)
    else:
        return obj.scale * obj.parameter.scale * obj.parameter.parameter * func(obj.linear @ variable)


def loss_to_cvxpy(obj, variable):
    match obj:
        case ml.LogisticLoss(y, X):
            return cp.sum(cp.logistic(-cp.multiply(y, X @ variable)))
        case ml.MSELoss(y, X):
            return cp.sum_squares(y - X @ variable) / y.numel()
        case _:
            raise RuntimeError("Unknown loss")


def transform_model_to_cvxpy(loss, regularizer, variable):
    return cp.Problem(
        loss_to_cvxpy(loss, variable) + regularizer_sum_to_cvxpy(regularizer, variable)
    )


class Jacobian(lo.LinearOperator):
    solution_func: Callable[[], torch.Tensor]
    loss: ml.Loss
    regularizer: ml.Sum | ml.Regularizer
    inverse_method: Literal[None, 'minres', 'cholesky'] = field(default=None)

    supports_operator_matrix = True

    def __init__(self, solution_func, loss, regularizer, inverse_method):
        self.solution_func = solution_func
        self.loss = loss
        self.regularizer = regularizer
        self.inverse_method = inverse_method
        
    @property
    def _shape(self):
        n = self.loss.y.shape[0]
        return (n, n)

    _diag: torch.Tensor = None

    @functools.cached_property
    def diag(self):
        return torch.diag(self @ torch.eye(self.shape[1]))

    def _matmul_impl(self, rhs):
        beta_hat = self.solution_func()
        y = utils.to_tensor(self.loss.y)
        X = utils.to_tensor(self.loss.X)
        y, y_hat, dloss_dy_hat, d2loss_dboth, d2loss_dy_hat2 = utils.compute_derivatives(self.loss.func, y, X @ beta_hat)

        mask = torch.ones_like(beta_hat.squeeze(), dtype=bool)

        constraints, hessians = unpack_regularizer(self.regularizer, mask, beta_hat)
        X_mask = X[:, mask]
        rhs_scaled =  (d2loss_dy_hat2[:, None] * rhs)

        if constraints is None and hessians is None:
            tilde_X = torch.sqrt(d2loss_dy_hat2)[:, None] * X_mask
            Q, R = torch.linalg.qr(tilde_X)
            return Q @ (Q.T @ rhs_scaled)
        elif constraints is None:
            kkt_rhs = X_mask.T @ rhs_scaled
            if hessians is None:
                tilde_X = torch.sqrt(d2loss_dy_hat2)[:, None] * X_mask
                _, R = torch.linalg.qr(tilde_X, mode='r')
            else:
                hessians_mask = hessians[mask, :][:, mask]
                P = X_mask.T @ (d2loss_dy_hat2[:, None] X_mask) + hessians_mask
                R = torch.linalg.cholesky(P, upper=True)
            v = torch.linalg.solve_triangular(
                R, torch.linalg.solve_triangular(
                    R.T, kkt_rhs, upper=False
                ), upper=True
            )
        else:
            constraints_mask = constraints[:, mask]
            n, m = constraints_mask.shape
            if n >= m:
                _, N = torch.linalg.qr(constraints_mask, mode='r')
            else:
                N = constraints_mask
            
            if hessians is None:
                tilde_X = torch.sqrt(d2loss_dy_hat2)[:, None] * X_mask
                _, P_R = torch.linalg.qr(tilde_X, mode='r')
            else:
                hessians_mask = hessians[mask, :][:, mask]
                P = X_mask.T @ (d2loss_dy_hat2[:, None] X_mask) + hessians_mask
                P_R = torch.linalg.cholesky(P, upper=True)
 
            S = self.D_nmask @ torch.linalg.solve_triangular(
                P_R, torch.linalg.solve_triangular(
                    P_R.T, kkt_rhs, upper=False
                ), upper=True
            )
            S_R = torch.linalg.cholesky(S, upper=True)
            NPinvRhs = N @ torch.linalg.solve_triangular(
                P_R, torch.linalg.solve_triangular(
                    P_R.T, kkt_rhs, upper=False
                ), upper=True
            )
            nu = torch.linalg.solve_triangular(
                S_R, torch.linalg.solve_triangular(
                    S_R.T, -NPinvRhs, upper=False
                ), upper=True
            )
            v = torch.linalg.solve_triangular(
                P_R, torch.linalg.solve_triangular(
                    P_R.T, kkt_rhs + N.T @ nu, upper=False
                ), upper=True
            )
            return X_mask @ v


def unpack_regularizer(regularizer, mask, beta_hat, epsilon=1e-6):
    """
    Modifies mask in place

    Returns a constraint matrix (or None) and a hessian term (or None)
    """

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
        case ml.SquareRegularizer(linear, scale, parameter):
            if linear is None:
                return None, torch.diag(
                    scale * parameter.value * torch.ones_like(mask, dtype=beta_hat.dtype)
                )
            elif isinstance(linear, list):
                diag = torch.zeros_like(mask, dtype=beta_hat.dtype)
                diag[linear] = scale * parameter.value
                return None, torch.diag(diag)
            else:
                A = utils.from_numpy(linear)
                return None, torch.diag(
                    scale * parameter.value * (A.mT @ A)
                )
        case ml.L1Regularizer(linear, _):
            if linear is None:
                mask[torch.abs(beta_hat) <= epsilon] = False
                return None, None
            elif isinstance(linear, list):
                mask[linear][torch.abs(beta_hat[linear]) <= epsilon] = False
                return None, None
            else:
                A = utils.from_numpy(linear)
                return A[torch.abs(A @ beta_hat) <= epsilon, :], None

        case ml.L2Regularizer(linear, scale, paramter):
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
            case ml.HuberRegularizer(linear, scale, parameter)

    return constraints, hessians


def transform_model_to_Jacobian(solution_func, loss, regularizer):
    return Jacobian(solution_func, loss, regularizer)
