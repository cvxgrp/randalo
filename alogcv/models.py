from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import (
    LinearOperator as ScipyLinearOperator,
    minres as scipy_minres,
)
import torch

from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso, LogisticRegression


import linops as lo
from linops import LinearOperator

from alogcv.minres import minres

# from linops.minres import minres

from . import utils


def sparse_safe_tensor(X):
    if sparse.issparse(X):
        return X.tocsc()
    else:
        return torch.tensor(X)


class ALOModel(ABC):
    def __init__(self):
        self.needs_compute_jac = True
        self.fitted = False

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model = self.get_new_model()
        self.model.fit(X, y)
        self.y_hat = self.model.predict(X)

        # Store data and compute derivatives using torch
        self._X = sparse_safe_tensor(X)
        (
            self._y,
            self._y_hat,
            self._dloss_dy_hat,
            self._d2loss_dboth,
            self._d2loss_dy_hat2,
        ) = utils.compute_derivatives(
            self.loss_fun, torch.tensor(y), torch.tensor(self.y_hat)
        )

        self._jac = None
        self.fitted = True
        self.needs_compute_jac = True
        return self

    def predict(self, X):
        self._fitted_check()
        return self.model.predict(X)

    def jac(self, device=None):
        self._fitted_check()
        if self.needs_compute_jac or (
            self._jac is not None and self._jac.device != device
        ):
            if device is None:
                device = self._X.device
            if not sparse.issparse(self._X):
                self._X = self._X.to(device)
            self._y = self._y.to(device)
            self._y_hat = self._y_hat.to(device)
            self._dloss_dy_hat = self._dloss_dy_hat.to(device)
            self._d2loss_dboth = self._d2loss_dboth.to(device)
            self._d2loss_dy_hat2 = self._d2loss_dy_hat2.to(device)
            self._jac = self._compute_jac(device=device)
            self.needs_compute_jac = False
        return self._jac

    @abstractmethod
    def _compute_jac(self):
        pass

    @abstractmethod
    def get_new_model(self):
        pass

    def _fitted_check(self):
        if not self.fitted:
            raise NotFittedError("Model has not yet been fit")

    @abstractstaticmethod
    def loss_fun(y, y_hat):
        pass


class LinearMixin(ABC):
    @property
    def coef_(self):
        self._fitted_check()
        return self.model.coef_


class LinearOperatorWrapper(LinearOperator):
    supports_operator_matrix = True

    def __init__(self, A, adjoint=None):
        self._shape = (A.shape[0], A.shape[0])
        self.device = A.device
        self.dtype = A.dtype
        self.A = A

        if adjoint is None:
            self._adjoint = LinearOperatorWrapper(A.t(), self)
        else:
            self._adjoint = adjoint

    def _matmul_impl(self, B):
        return self.A @ B


class loATD1APlusD2(LinearOperator):
    """
    Linear operator representing A.T @ D1 @ A + D2,
    where A is a linear operator or matrix and D1
    and D2 are diagonal matrices."""

    supports_operator_matrix = True

    def __init__(self, A, D1, D2):
        self._shape = (A.shape[1], A.shape[1])
        self.dtype = A.dtype
        self.A = A
        self.D1 = D1
        self.D2 = D2

    def _matmul_impl(self, v):
        return self.A.T @ (self.D1[:, None] * (self.A @ v)) + self.D2[:, None] * v


class ATD1APlusD2(ScipyLinearOperator):
    """Linear operator representing A.T @ D1 @ A + D2, where A is a linear operator and D1 and D2 are diagonal matrices."""

    def __init__(self, A, D1, D2):
        self.shape = (A.shape[1], A.shape[1])
        self.dtype = A.dtype
        self.A = A
        self.D1 = D1
        self.D2 = D2

    def _matvec(self, v):
        return self.A.T @ (self.D1 @ (self.A @ v)) + self.D2 @ v

    def _matmat(self, V):
        return self.A.T @ (self.D1 @ (self.A @ V)) + self.D2 @ V

    def _adjoint(self):
        return self


class SeparableRegularizerJacobian(LinearOperator):

    # supports_operator_matrix = True
    @property
    def supports_operator_matrix(self):
        return not self.issparse

    def __init__(
        self,
        X,
        loss_hessian_diag,
        loss_dy_dy_hat_diag,
        reg_hessian_diag,
        use_direct_method=None,
    ):
        n = X.shape[0]
        self._shape = (n, n)

        self._direct = use_direct_method if use_direct_method is not None else n < 2500

        if sparse.issparse(X):
            X = X.tocsc()
            self.issparse = True
            self.device = "cpu"
            self._direct = False
        else:
            self.issparse = False
            self.device = X.device
        self.dtype = X.dtype
        self.loss_hessian_diag = loss_hessian_diag
        self.loss_dy_dy_hat_diag_ = loss_dy_dy_hat_diag

        mask = torch.isfinite(reg_hessian_diag)
        self.X_mask = X[:, mask]

        # sparse X case:
        if self.issparse:
            assert not self._direct, "Direct solver for sparse data isn't supported"
            D1 = sparse.diags(loss_hessian_diag.numpy())
            D2 = sparse.diags(reg_hessian_diag[mask].numpy())
            self.H = ATD1APlusD2(self.X_mask, D1, D2)
        # dense X case:
        elif self._direct:
            H = self.X_mask.T @ (loss_hessian_diag[:, None] * self.X_mask) + torch.diag(
                reg_hessian_diag[mask]
            )
            self.LD, self.pivots = torch.linalg.ldl_factor(H)
        else:
            self.H = loATD1APlusD2(
                self.X_mask, loss_hessian_diag, reg_hessian_diag[mask]
            )

    def _matmul_impl(self, A):
        if A.ndim == 1:
            A = A[:, None]
            need_squeeze = True
        else:
            need_squeeze = False

        A = -self.loss_dy_dy_hat_diag_[:, None] * A

        # dense X case:
        if not self.issparse:
            B = self.X_mask.T @ A
            if self._direct:
                Z = torch.linalg.ldl_solve(self.LD, self.pivots, B)
            else:
                Z = minres(self.H, B, tol=1e-2)
        # sparse X case:
        else:
            B = self.X_mask.T @ A.detach().numpy()
            Z = scipy_minres(self.H, B)[0]
            need_squeeze = False

        if need_squeeze:
            Z = Z[..., 0]

        return torch.as_tensor(self.X_mask @ Z)

    def todense(self):

        # dense X and direct solve case:
        if not self.issparse and self._direct:
            return -self.X_mask @ torch.linalg.ldl_solve(
                self.LD, self.pivots, self.X_mask.T * self.loss_dy_dy_hat_diag_[None, :]
            )

        # sparse X or indirect solve case:
        return self @ torch.eye(self.shape[0])

    @property
    def diag(self):
        return torch.diag(self.todense())


class LinearSeparableRegularizerJacobian(LinearOperator):
    supports_operator_matrix = True

    def __init__(self, X, d2loss_dboth, D, loss_hessian_diag, reg_hessian_diag):
        self._shape = (X.shape[0], X.shape[0])
        self.device = X.device
        self.dtype = X.dtype
        self.d2loss_dboth = d2loss_dboth
        self.loss_hessian_diag = loss_hessian_diag

        self.X = X
        mask = torch.isfinite(reg_hessian_diag)
        self.D_mask = D[mask, :]
        self.reg_hessian_diag_mask = reg_hessian_diag[mask]

        if torch.linalg.vector_norm(self.reg_hessian_diag_mask) <= 1e-9:
            H_sqrt = torch.sqrt(loss_hessian_diag)[:, None] * self.X
            _, self.H_R = torch.linalg.qr(H_sqrt, mode="r")
        else:
            H = self.X.T @ (loss_hessian_diag[:, None] * self.X) + self.D_mask.T @ (
                self.reg_hessian_diag_mask[:, None] * self.D_mask
            )
            self.H_R = torch.linalg.cholesky(H, upper=True)
        self.D_nmask = D[~mask, :]

        M = self.D_nmask @ self._Hinv(self.D_nmask.T)
        self.M_L = torch.linalg.cholesky(M)

    def _Hinv(self, V):
        return torch.linalg.solve_triangular(
            self.H_R,
            torch.linalg.solve_triangular(self.H_R.T, V, upper=False),
            upper=True,
        )

    def _Minv(self, V):
        return torch.linalg.solve_triangular(
            self.M_L.T,
            torch.linalg.solve_triangular(self.M_L, V, upper=False),
            upper=True,
        )

    def _matmul_impl(self, A):
        if A.ndim == 1:
            A = A[:, None]
            need_squeeze = True
        else:
            need_squeeze = False

        # Boyd and Vandenberghe, Convex Optimization, p. 545
        G = -self.X.T @ (self.d2loss_dboth[:, None] * A)
        DHinvG = self.D_nmask @ self._Hinv(G)
        lagrange_multiples = self._Minv(-DHinvG)
        Z = -self._Hinv(G + self.D_nmask.T @ lagrange_multiples)

        if need_squeeze:
            Z = Z[..., 0]

        return self.X @ Z

    @property
    def diag(self):
        return torch.diag(self @ torch.eye(self.shape[0]))


class RandomForestRegressorJacobian(LinearOperator):

    supports_operator_matrix = True

    def __init__(self, forest, X, dtype, device="cpu"):

        self._shape = (X.shape[0], X.shape[0])
        self.dtype = dtype
        self.device = device

        J = np.zeros((X.shape[0], X.shape[0]))
        for est in forest.estimators_:
            tree = est.tree_
            S = sparse.csr_array(
                (
                    np.ones(X.shape[0]),
                    tree.apply(X.astype(np.float32)),
                    np.arange(X.shape[0] + 1),
                ),
                shape=(X.shape[0], tree.node_count),
            )
            denom = S.sum(axis=0)[None, :]
            # Avoid division by zero
            denom[denom == 0] = 1
            J += (S / denom) @ S.T
        J /= len(forest.estimators_)

        self.J = sparse.csr_array(J)

    def _matmul_impl(self, A):
        return torch.as_tensor(self.J @ A.detach().numpy())

    def todense(self):
        return self.J.toarray()

    @property
    def diag(self):
        return torch.as_tensor(self.J.diagonal())


class SeparableRegularizerMixin(ABC):
    @property
    @abstractmethod
    def reg_hessian_diag_(self):
        pass

    def _compute_jac(self, device=None):

        return SeparableRegularizerJacobian(
            self._X,
            self._d2loss_dy_hat2,
            self._d2loss_dboth,
            self.reg_hessian_diag_.to(device),
            self._direct,
        )


class LinearSeparableRegularizerMixin(ABC):
    @property
    @abstractmethod
    def loss_hessian_diag_(self):
        pass

    @property
    @abstractmethod
    def reg_hessian_diag_(self):
        pass

    @abstractmethod
    def generate_D_from_X(self, X):
        pass

    def _compute_jac(self, device=None):
        return LinearSeparableRegularizerJacobian(
            self._X,
            self._d2loss_dy_hat2,
            self.generate_D_from_X(self.X),
            self._d2loss_dboth,
            self.reg_hessian_diag_.to(device),
        )


class LassoModel(LinearMixin, SeparableRegularizerMixin, ALOModel):
    """Lasso model for ALO computation

    The optimization objective is given by
    ```
    1 / (2 * n) * ||y - Xw||^2_2 + lamda * ||w||_1
    ```

    """

    def __init__(
        self,
        lamda,
        sklearn_lasso_kwargs={},
        direct: Optional[bool] = None,
    ):
        super().__init__()
        self.lamda = lamda
        self.sklearn_lasso_kwargs = sklearn_lasso_kwargs
        self._direct = direct

    def get_new_model(self):
        kwargs = self.sklearn_lasso_kwargs.copy()
        kwargs["alpha"] = self.lamda
        kwargs["fit_intercept"] = False
        return Lasso(**kwargs)

    @property
    def reg_hessian_diag_(self):
        self._fitted_check()
        hess = np.zeros(self.X.shape[1])
        hess[self.model.coef_ == 0] = float("inf")
        return torch.tensor(hess)

    @staticmethod
    def loss_fun(y, y_hat):
        return (y - y_hat) ** 2 / 2


class FirstDifferenceModel(LinearMixin, LinearSeparableRegularizerMixin, ALOModel):
    """First Difference Model for ALO computation

    The optimization objective is given by
    ```
    1 / (2 * n)) * ||y - Xw||^2_2 + lamda * ||D w||_1
    ```
    where D = is torch.diff
    """

    def __init__(
        self,
        lamda,
        cvxpy_kwargs={},
    ):
        super().__init__()
        self.lamda = lamda
        self.cvxpy_kwargs = cvxpy_kwargs

    def get_new_model(self):
        class C:
            coef_ = None

            def fit(s, X, y):
                import cvxpy as cp

                w = cp.Variable(X.shape[1])
                t = cp.Variable()
                prob = cp.Problem(
                    cp.Minimize(
                        1 / (2 * X.shape[0]) * cp.sum_squares(y - X @ w)
                        + self.lamda * cp.norm(cp.diff(w), 1)
                    ),
                )
                prob.solve(cp.CLARABEL, **self.cvxpy_kwargs)
                s.coef_ = w.value

            def predict(s, x):
                return x @ s.coef_

        return C()

    @property
    def reg_hessian_diag_(self):
        self._fitted_check()
        hess = np.zeros(self.X.shape[1] - 1)
        hess[np.abs(np.diff(self.model.coef_)) <= 1e-6] = float("inf")
        return torch.tensor(hess)

    @staticmethod
    def loss_fun(y, y_hat):
        return (y - y_hat) ** 2 / 2

    def generate_D_from_X(self, X):
        return torch.diff(torch.eye(X.shape[1]), axis=0)


class DecisionFunctionModelWrapper(LinearMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.decision_function(X)

    @property
    def coef_(self):
        return self.model.coef_.squeeze()


class LogisticModel(LinearMixin, SeparableRegularizerMixin, ALOModel):
    """Logistic model for ALO computation

    The optimization objective is given by
    ```
    1 / n * sum_i log(1 + exp(-y_i * x_i^T w)) + lamda / p * ||w||_p^p
    ```
    where the norm is either the l1 norm or the l2 norm squared as
    specified by the `penalty` parameter in `sklearn_logistic_kwargs`.

    """

    def __init__(
        self,
        lamda,
        sklearn_logistic_kwargs={},
        direct: Optional[bool] = None,
    ):
        super().__init__()
        self.lamda = lamda
        self.sklearn_logistic_kwargs = sklearn_logistic_kwargs
        self._direct = direct

    def get_new_model(self):
        kwargs = self.sklearn_logistic_kwargs.copy()
        kwargs["C"] = 1 / self.lamda
        return DecisionFunctionModelWrapper(LogisticRegression(**kwargs))

    @property
    def reg_hessian_diag_(self):
        self._fitted_check()

        if self.sklearn_logistic_kwargs["penalty"] == "l1":
            hess = np.zeros(self.X.shape[1])
            hess[self.coef_ == 0] = float("inf")
        else:
            hess = np.ones(self.X.shape[1]) * self.lamda

        return torch.tensor(hess)

    @staticmethod
    def loss_fun(y, y_hat):
        return torch.log(1 + torch.exp(-y * y_hat))


class RandomForestRegressorModel(ALOModel):

    def __init__(self, sklearn_rf_kwargs):
        super().__init__()
        self.sklearn_rf_kwargs = sklearn_rf_kwargs
        self.sklearn_rf_kwargs["bootstrap"] = True
        self.sklearn_rf_kwargs["oob_score"] = True

    def get_new_model(self):
        return RandomForestRegressor(**self.sklearn_rf_kwargs)

    def _compute_jac(self, device=None):
        return RandomForestRegressorJacobian(
            self.model, self._X.detach().numpy(), self._X.dtype, self._X.device
        )

    def oob_prediction(self):
        self._fitted_check()
        return self.model.oob_prediction_

    @staticmethod
    def loss_fun(y, y_hat):
        return (y - y_hat) ** 2 / 2
