import functools
from typing import Callable, Literal
from dataclasses import dataclass, field

import numpy as np
import linops as lo
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
    solution_func: Callable[[], torch.Tensor]
    loss: ml.Loss
    regularizer: ml.Sum | ml.Regularizer
    inverse_method: Literal[None, "minres", "cholesky"]

    supports_operator_matrix = True

    def __init__(self, y, X, solution_func, loss, regularizer, inverse_method=None):
        super().__init__()
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

    # @functools.cached_property
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

        constraints, hessians, mask = \
                self.regularizer.get_constraint_hessian_mask(beta_hat)
        if mask is not None:
            X_mask = X[:, mask]
        else:
            X_mask = X
        rhs_scaled = -d2loss_dboth[:, None] * rhs

        if constraints is None and hessians is None:
            if torch.any(d2loss_dy_hat2 == 0):
                # The weighted-QR identity below contains D^{-1/2}. A zero
                # sample weight makes that expression undefined even though
                # the original normal equations remain well-defined.
                P = X_mask.T @ (d2loss_dy_hat2[:, None] * X_mask)
                v = torch.linalg.lstsq(P, X_mask.T @ rhs_scaled).solution
                out = X_mask @ v
                return out if not needs_squeeze else out.squeeze(-1)

            work_dtype = X_mask.dtype
            sqrt_d2loss_dy_hat2 = torch.sqrt(d2loss_dy_hat2)
            dynamic_range = (
                sqrt_d2loss_dy_hat2.max() / sqrt_d2loss_dy_hat2.min()
            )
            if X_mask.dtype == torch.float32 and dynamic_range > 1e4:
                # Nearly separable unregularized logistic fits can have loss
                # curvature spanning many orders of magnitude. Float32 QR is
                # not accurate enough after the D^{-1/2} rescaling.
                work_dtype = torch.float64

            X_work = X_mask.to(work_dtype)
            rhs_work = rhs_scaled.to(work_dtype)
            sqrt_d2loss_dy_hat2 = sqrt_d2loss_dy_hat2.to(work_dtype)[:, None]
            tilde_X = sqrt_d2loss_dy_hat2 * X_work
            Q, _ = torch.linalg.qr(tilde_X)
            out = (
                Q @ (Q.T @ (rhs_work / sqrt_d2loss_dy_hat2))
            ) / sqrt_d2loss_dy_hat2
            out = out.to(X_mask.dtype)
            return out if not needs_squeeze else out.squeeze(-1)
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
