from abc import ABC, abstractmethod, abstractstaticmethod

from sklearn.linear_model import Lasso
from sklearn.exceptions import NotFittedError

import torch

from linops import LinearOperator


class ALOModel(ABC):

    def __init__(self):
        self.needs_compute_jac = True
        self.fitted = False

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model = self.get_new_model()
        self.model.fit(X, y)

        self._jac = None
        self.fitted = True
        self.needs_compute_jac = True

    def predict(self, X):
        self._fitted_check()
        return self.model.predict(X)

    def jac(self, device=None):
        self._fitted_check()
        if self.needs_compute_jac or (
            self._jac is not None and self._jac.device != device
        ):
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

    def __init__(self, X, loss_hessian_diag, reg_hessian_diag):

        self._shape = (X.shape[0], X.shape[0])
        self.device = X.device
        self.dtype = X.dtype
        self.loss_hessian_diag = loss_hessian_diag

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

        A = self.loss_hessian_diag[:, None] * A

        Z = torch.linalg.ldl_solve(self.LD, self.pivots, self.X_mask.T @ A)

        if need_squeeze:
            Z = Z[..., 0]

        return self.X_mask @ Z

    def __matmul__(self, A):
        return self._matmul_impl(A)

    def _adjoint(self):
        return self

    @property
    def diag(self):
        return self.X_mask @ torch.linalg.ldl_solve(self.LD, self.pivots, self.X_mask.T)


class SeparableRegularizerMixin(ABC):

    @property
    @abstractmethod
    def loss_hessian_diag_(self):
        pass

    @property
    @abstractmethod
    def reg_hessian_diag_(self):
        pass

    def _compute_jac(self, device=None):
        return SeparableRegularizerJacobian(
            torch.tensor(self.X, device=device),
            torch.tensor(self.loss_hessian_diag_, device=device),
            torch.tensor(self.reg_hessian_diag_, device=device),
        )


class LassoModel(LinearMixin, SeparableRegularizerMixin, ALOModel):
    """Lasso model for ALO computation

    The optimization objective is given by
    ```
    1 / (2 * n)) * ||y - Xw||^2_2 + lamda * ||w||_1
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
    def loss_hessian_diag_(self):
        self._fitted_check()
        return torch.ones(self.X.shape[0]) / self.X.shape[0]

    @property
    def reg_hessian_diag_(self):
        self._fitted_check()
        hess = torch.zeros(self.X.shape[1])
        hess[self.model.coef_ == 0] = float("inf")
        return hess

    @staticmethod
    def loss_fun(y, y_hat):
        return (y - y_hat) ** 2 / 2
