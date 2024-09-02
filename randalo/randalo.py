import linops as lo
import numpy as np
import sklearn
import torch

from . import modeling_layer as ml
from . import util


class RandALO(object):

    def __init__(
        self,
        loss: ml.Loss = None,
        jac: lo.LinearOperator = None,
        y: torch.Tensor | np.ndarray = None,
        y_hat: torch.Tensor | np.ndarray = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        if loss is None:
            raise ValueError("loss function must be provided")
        self._loss = loss

        if jac is None:
            raise ValueError("Jacobian operator must be provided")
        self._jac = jac

        if y is None:
            raise ValueError("label values must be provided")
        y = util.to_tensor(y, dtype=dtype, device=device)

        if y_hat is None:
            raise ValueError("predicted values must be provided")
        y_hat = util.to_tensor(y_hat, dtype=dtype, device=device)

        self.dtype = dtype
        self.device = device

        # check dtypes and devices
        assert self._jac.dtype == self.dtype
        assert self._jac.device == self.device

        # compute derivatives of loss function
        (
            self._y,
            self._y_hat,
            self._dloss_dy_hat,
            self._d2loss_dboth,
            self._d2loss_dy_hat2,
        ) = util.compute_derivatives(self._loss, y, y_hat)

        # precompute some quantities for working with generic form
        # a = dloss_dy_hat
        # b = d2loss_dy_hat2
        # c = d2loss_dboth
        self._a_c = self.dloss_dy_hat / self.d2loss_dy_hat2
        self._c_b = self.d2loss_dy_hat2 / self.d2loss_dboth

    @classmethod
    def from_sklearn(cls, model, X, y):

        sklearn.utils.validation.check_is_fitted(model)

        pass
