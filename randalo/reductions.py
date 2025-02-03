import functools
from typing import Callable, Literal
from dataclasses import dataclass, field

import scipy
import numpy as np
import linops as lo
from linops.minres import minres
import torch

from . import modeling_layer as ml
from . import utils

def gen_cvxpy_jacobian(loss, regularizer, X, variable, y, inversion_method=None):
    prob = transform_model_to_cvxpy(loss, regularizer, X, y, variable)
    J = Jacobian(y, X, lambda: variable.value, loss, regularizer, inversion_method=None)
    return prob, J

def transform_model_to_cvxpy(loss, regularizer, X, y, variable):
    import cvxpy as cp
    return cp.Problem(
        cp.Minimize(
            loss.to_cvxpy(y, X @ regularizer) +
            regularizer.to_cvxpy(variable)
        )
    )


class Jacobian(lo.LinearOperator):
    solution_func: Callable[[], torch.Tensor] | Callable[[], tuple[torch.Tensor, torch.Tensor]]
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
        self.X = lo.aslinearoperator(X)

        self._adjoint = self  # Not actually symmetric

    @property
    def _shape(self):
        n = self.y.shape[0]
        return (n, n)

    _diag: torch.Tensor = None

    # @functools.cached_property
    def diag(self):
        return torch.diag(self @ torch.eye(self.shape[1]))

    def _matmul_impl(self, rhs):
        y = self.y
        X = self.X

        rhs = utils.to_tensor(rhs)
        needs_squeeze = False
        if len(rhs.shape) == 1:
            rhs = rhs.unsqueeze(-1)
            needs_squeeze = True

        solution = self.solution_func()
        if isinstance(solution, scipy.sparse.csr_matrix):
            beta_hat = utils.to_tensor(solution.data)
            mask_0 = utils.to_tensor(solution.indices, dtype=torch.int)
            if solution.data.shape == (0,):
                return torch.zeros_like(rhs).squeeze() if needs_squeeze else torch.zeros_like(rhs)
            X = X[:, mask_0]
            constraints, hessians, mask = \
                    self.regularizer.get_constraint_hessian_mask_sparse(
                            beta_hat, mask_0, X.shape[1])
        else:
            beta_hat = utils.to_tensor(solution)

            constraints, hessians, mask = \
                    self.regularizer.get_constraint_hessian_mask(beta_hat)

        if mask is not None:
            X_mask = X[:, mask]
        else:
            X_mask = X

        _, _, _, d2loss_dboth, d2loss_dy_hat2 = utils.compute_derivatives(
            self.loss, y, X @ beta_hat
        )

        rhs_scaled = -d2loss_dboth[:, None] * rhs

        # TODO: Split cholesky/minres code paths into seperate ones
        if self.inverse_method == 'minres':
            if constraints is None and hessians is None:
                import time
                print("Line 1", time.monotonic(), flush=True)
                sqrt_d2loss_dy_hat2 = torch.sqrt(d2loss_dy_hat2)[:, None]
                print("Line 2", time.monotonic(), flush=True)
                tilde_X = sqrt_d2loss_dy_hat2 * X_mask
                print("Line 3", time.monotonic(), flush=True)
                p1 = (rhs_scaled / sqrt_d2loss_dy_hat2)
                print("Line 4", time.monotonic(), flush=True)
                p2 = X.T @ p1
                print("Line 5", time.monotonic(), flush=True)
                p3 = X.T @ X
                print("Line 6", time.monotonic(), flush=True)
                p4 = minres(p3, p2)
                print("Line 7", time.monotonic(), flush=True)
                p5 = X @ p4
                print("Line 8", time.monotonic(), flush=True)
                return (p5 / sqrt_d2loss_dy_hat2).to(rhs.dtype)
            else:
                raise NotImplementedError()

        elif constraints is None and hessians is None:
            sqrt_d2loss_dy_hat2 = torch.sqrt(d2loss_dy_hat2)[:, None]
            tilde_X = sqrt_d2loss_dy_hat2 * X_mask
            Q, _ = torch.linalg.qr(tilde_X)
            return (
                Q @ (Q.T @ (rhs_scaled / sqrt_d2loss_dy_hat2))
            ) / sqrt_d2loss_dy_hat2
        elif constraints is None:
            # TODO: double check this doesn't need additional scaling
            kkt_rhs = X_mask.T @ rhs_scaled
            if mask is not None:
                hessians_mask = hessians[mask, :][:, mask]
            else:
                hessians_mask = hessians
            P = X_mask.T @ (d2loss_dy_hat2[:, None] * X_mask) + hessians_mask
            R = torch.linalg.cholesky(P, upper=True)
            v = torch.linalg.solve_triangular(
                R, torch.linalg.solve_triangular(R.T, kkt_rhs, upper=False), upper=True
            )
        else:
            # TODO: double check this doesn't need additional scaling
            if mask is not None:
                constraints_mask = constraints[:, mask]
            else:
                constraints_mask = constraints
            n, m = constraints_mask.shape
            if n >= m:
                _, N = torch.linalg.qr(constraints_mask, mode="r")
            else:
                N = constraints_mask

            if hessians is None:
                tilde_X = torch.sqrt(d2loss_dy_hat2)[:, None] * X_mask
                _, P_R = torch.linalg.qr(tilde_X, mode="r")
            else:
                if mask is not None:
                    hessians_mask = hessians[mask, :][:, mask]
                else:
                    hessian_mask = hessians
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
