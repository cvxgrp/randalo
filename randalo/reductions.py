import functools
from typing import Callable, Literal
from dataclasses import dataclass, field
import warnings

import numpy as np
import linops as lo
import scipy.sparse
import scipy.sparse.linalg
import torch

from . import modeling_layer as ml
from . import utils


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _sparse_regularizer_parts(regularizer, beta_hat, epsilon=1e-6):
    """Return sparse constraint and Hessian data without dense diagonals."""
    if isinstance(regularizer, ml.Sum):
        constraints = []
        hessians = []
        mask = torch.ones_like(beta_hat, dtype=bool)
        has_mask = False
        for expression in regularizer.exprs:
            constraint, hessian, expression_mask = _sparse_regularizer_parts(
                expression, beta_hat, epsilon
            )
            if constraint is not None:
                constraints.append(constraint)
            if hessian is not None:
                hessians.append(hessian)
            if expression_mask is not None:
                mask &= expression_mask
                has_mask = True

        constraint = (
            scipy.sparse.vstack(constraints, format="csr")
            if constraints
            else None
        )
        hessian = sum(hessians[1:], start=hessians[0]) if hessians else None
        return constraint, hessian, mask if has_mask else None

    if isinstance(regularizer, ml.SquareRegularizer) and (
        regularizer.linear is None or isinstance(regularizer.linear, list)
    ):
        scale = regularizer._scale()
        if scale == 0.0:
            return None, None, None
        diagonal = np.zeros(beta_hat.numel())
        if regularizer.linear is None:
            diagonal.fill(2 * scale)
        else:
            diagonal[regularizer.linear] = 2 * scale
        return None, scipy.sparse.diags(diagonal, format="csr"), None

    constraint, hessian, mask = regularizer.get_constraint_hessian_mask(
        beta_hat, epsilon
    )
    if constraint is not None:
        constraint = scipy.sparse.csr_matrix(_to_numpy(constraint))
    if hessian is not None:
        hessian = scipy.sparse.csr_matrix(_to_numpy(hessian))
    return constraint, hessian, mask


def _solve_sparse_system(matrix, rhs):
    matrix = matrix.tocsc()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", scipy.sparse.linalg.MatrixRankWarning)
            solution = scipy.sparse.linalg.spsolve(matrix, rhs)
        if np.all(np.isfinite(solution)):
            return np.asarray(solution).reshape(matrix.shape[1], -1)
    except (scipy.sparse.linalg.MatrixRankWarning, RuntimeError):
        pass

    return np.column_stack(
        [
            scipy.sparse.linalg.lsmr(
                matrix, column, atol=1e-10, btol=1e-10
            )[0]
            for column in rhs.T
        ]
    )


def _solve_sparse_least_squares(X, curvature, rhs):
    nonzero = curvature > 0
    if not np.any(nonzero):
        return np.zeros((X.shape[1], rhs.shape[1]))

    square_root = np.sqrt(curvature[nonzero])
    weighted_X = X[nonzero].multiply(square_root[:, None])
    weighted_rhs = np.asarray(
        rhs[nonzero] / square_root[:, None], dtype=weighted_X.dtype
    )
    return np.column_stack(
        [
            scipy.sparse.linalg.lsmr(
                weighted_X, column, atol=1e-10, btol=1e-10
            )[0]
            for column in weighted_rhs.T
        ]
    )

def gen_cvxpy_jacobian(loss, regularizer, X, variable, y, inversion_method=None):
    prob = transform_model_to_cvxpy(loss, regularizer, X, y, variable)
    J = Jacobian(
        y,
        X,
        lambda: variable.value,
        loss,
        regularizer,
        inverse_method=inversion_method,
    )
    return prob, J

def transform_model_to_cvxpy(loss, regularizer, X, y, variable):
    import cvxpy as cp
    return cp.Problem(
        cp.Minimize(
            loss.to_cvxpy(y, X @ variable) +
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
        if scipy.sparse.issparse(X):
            dtype = np.result_type(X.dtype, np.float32)
            self.X = scipy.sparse.csr_matrix(X, dtype=dtype)
        else:
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
        if scipy.sparse.issparse(self.X):
            return self._sparse_matmul(rhs, needs_squeeze)

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

    def _sparse_matmul(self, rhs, needs_squeeze):
        beta_hat = utils.to_tensor(self.solution_func())
        beta_numpy = _to_numpy(beta_hat)
        y_hat = torch.as_tensor(
            np.asarray(self.X @ beta_numpy).reshape(-1), dtype=self.y.dtype
        )
        _, _, _, d2loss_dboth, d2loss_dy_hat2 = utils.compute_derivatives(
            self.loss, self.y, y_hat
        )

        constraints, hessians, mask = _sparse_regularizer_parts(
            self.regularizer, beta_hat
        )
        if mask is None:
            mask_numpy = None
            X_mask = self.X
        else:
            mask_numpy = _to_numpy(mask)
            X_mask = self.X[:, mask_numpy]

        rhs_scaled = _to_numpy(-d2loss_dboth[:, None] * rhs)
        curvature = _to_numpy(d2loss_dy_hat2)
        if constraints is None and hessians is None:
            solution = _solve_sparse_least_squares(
                X_mask, curvature, rhs_scaled
            )
        else:
            weighted_X = X_mask.multiply(curvature[:, None])
            system = X_mask.T @ weighted_X
            if hessians is not None:
                if mask_numpy is not None:
                    hessians = hessians[mask_numpy][:, mask_numpy]
                system = system + hessians

            system_rhs = np.asarray(X_mask.T @ rhs_scaled)
            if constraints is not None:
                if mask_numpy is not None:
                    constraints = constraints[:, mask_numpy]
                n_constraints = constraints.shape[0]
                system = scipy.sparse.bmat(
                    [
                        [system, constraints.T],
                        [
                            constraints,
                            scipy.sparse.csr_matrix(
                                (n_constraints, n_constraints)
                            ),
                        ],
                    ],
                    format="csc",
                )
                system_rhs = np.vstack(
                    (
                        system_rhs,
                        np.zeros((n_constraints, system_rhs.shape[1])),
                    )
                )

            solution = _solve_sparse_system(system, system_rhs)
            solution = solution[: X_mask.shape[1]]

        out = torch.as_tensor(
            np.asarray(X_mask @ solution), dtype=self.y.dtype
        )
        return out if not needs_squeeze else out.squeeze(-1)
