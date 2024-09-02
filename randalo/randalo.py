import numpy as np
import torch

from linops import LinearOperator

from .modeling_layer import Loss, Jacobian
from .util import to_tensor


class RandALO(object):

    def __init__(
        self,
        loss: Loss = None,
        jac: LinearOperator = None,
        y: torch.Tensor | np.ndarray = None,
        y_hat: torch.Tensor | np.ndarray = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        if loss is None:
            raise ValueError("loss function must be provided")
        self.loss = loss

        if jac is None:
            raise ValueError("Jacobian operator must be provided")
        self.jac = jac

        if y is None:
            raise ValueError("label values must be provided")
        self.y = to_tensor(y, dtype=dtype, device=device)

        if y_hat is None:
            raise ValueError("predicted values must be provided")
        self.y_hat = to_tensor(y_hat, dtype=dtype, device=device)

        self.dtype = dtype
        self.device = device

        # check dtypes and devices
        assert self.jac.dtype == self.dtype
        assert self.jac.device == self.device

    @classmethod
    def from_sklearn(cls, model, X, y):
        pass
