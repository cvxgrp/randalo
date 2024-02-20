from abc import ABC, abstractmethod, abstractstaticmethod

import numpy as np
import torch

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.exceptions import NotFittedError

from linops import LinearOperator

from . import utils


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
        self._X = torch.tensor(X)
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


class SeparableRegularizerJacobian(LinearOperator):
    supports_operator_matrix = True

    def __init__(self, X, loss_hessian_diag, loss_dy_dy_hat_diag, reg_hessian_diag):
        self._shape = (X.shape[0], X.shape[0])
        self.device = X.device
        self.dtype = X.dtype
        self.loss_hessian_diag = loss_hessian_diag
        self.loss_dy_dy_hat_diag_ = loss_dy_dy_hat_diag

        mask = torch.isfinite(reg_hessian_diag)
        self.X_mask = X[:, mask]
        H = self.X_mask.T @ (loss_hessian_diag[:, None] * self.X_mask) + torch.diag(
            reg_hessian_diag[mask]
        )
        self.LD, self.pivots = torch.linalg.ldl_factor(H)

    def _matmul_impl(self, A):
        if A.ndim == 1:
            A = A[:, None]
            need_squeeze = True
        else:
            need_squeeze = False

        A = -self.loss_dy_dy_hat_diag_[:, None] * A

        Z = torch.linalg.ldl_solve(self.LD, self.pivots, self.X_mask.T @ A)

        if need_squeeze:
            Z = Z[..., 0]

        return self.X_mask @ Z

    @property
    def diag(self):
        return torch.diag(
            -self.X_mask
            @ torch.linalg.ldl_solve(
                self.LD, self.pivots, self.X_mask.T * self.loss_dy_dy_hat_diag_[None, :]
            )
        )


class LinearSeparableRegularizerJacobian(LinearOperator):
    supports_operator_matrix = True

    def __init__(self, X, D, loss_hessian_diag, reg_hessian_diag):
        self._shape = (X.shape[0], X.shape[0])
        self.device = X.device
        self.dtype = X.dtype
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
        G = -self.X.T @ (self.loss_hessian_diag[:, None] * A)
        DHinvG = self.D_nmask @ self._Hinv(G)
        lagrange_multiples = self._Minv(-DHinvG)
        Z = -self._Hinv(G + self.D_nmask.T @ lagrange_multiples)

        if need_squeeze:
            Z = Z[..., 0]

        return self.X @ Z

    @property
    def diag(self):
        return torch.diag(self @ torch.eye(self.shape[0]))


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
            torch.tensor(self.X, device=device),
            torch.tensor(self.generate_D_from_X(self.X), device=device),
            torch.tensor(self.loss_hessian_diag_, device=device),
            torch.tensor(self.reg_hessian_diag_, device=device),
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
    ):
        super().__init__()
        self.lamda = lamda
        self.sklearn_lasso_kwargs = sklearn_lasso_kwargs

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
                        1 / (2 * X.shape[0]) * cp.sum_squares(y - X @ w) + self.lamda * cp.norm(cp.diff(w), 1)
                    ),
                )
                prob.solve(cp.CLARABEL, **self.cvxpy_kwargs)
                s.coef_ = w.value

            def predict(s, x):
                return x @ s.coef_

        return C()

    @property
    def loss_hessian_diag_(self):
        self._fitted_check()
        return np.ones(self.X.shape[0]) / self.X.shape[0]

    @property
    def reg_hessian_diag_(self):
        self._fitted_check()
        hess = np.zeros(self.X.shape[1] - 1)
        hess[np.abs(np.diff(self.model.coef_)) <= 1e-6] = float("inf")
        return hess

    @staticmethod
    def loss_fun(y, y_hat):
        return (y - y_hat) ** 2 / 2

    def generate_D_from_X(self, X):
        return np.diff(np.eye(X.shape[1]), axis=0)


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
    1 / n * sum_i log(1 + exp(-y_i * x_i^T w)) + lamda * ||w||_1
    ```

    """

    def __init__(
        self,
        lamda,
        sklearn_logistic_kwargs={},
    ):
        super().__init__()
        self.lamda = lamda
        self.sklearn_logistic_kwargs = sklearn_logistic_kwargs

    def get_new_model(self):
        kwargs = self.sklearn_logistic_kwargs.copy()
        kwargs["C"] = 1 / self.lamda
        return DecisionFunctionModelWrapper(LogisticRegression(**kwargs))

    @property
    def reg_hessian_diag_(self):
        self._fitted_check()
        hess = np.zeros(self.X.shape[1])
        hess[self.coef_ == 0] = float("inf")
        return torch.tensor(hess)

    @staticmethod
    def loss_fun(y, y_hat):
        return torch.log(1 + torch.exp(-y * y_hat))
