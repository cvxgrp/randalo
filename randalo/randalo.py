from .metrics import squared_error

import numpy as np
import torch

from linops import LinearOperator


class RandALO(object):

    def __init__(
        self,
        loss: callable = None,
        jac: LinearOperator = None,
        y: torch.Tensor | np.ndarray = None,
        y_hat: torch.Tensor | np.ndarray = None,
    ):
        if loss is None:
            raise ValueError("loss function must be provided")
        self.loss = loss

        if jac is None:
            raise ValueError("Jacobian operator must be provided")
        self.jac = jac

        if y is None:
            raise ValueError("label values must be provided")
        self.y = torch.tensor(y) if isinstance(y, np.ndarray) else y

        if y_hat is None:
            raise ValueError("predicted values must be provided")
        self.y_hat = torch.tensor(y_hat) if isinstance(y_hat, np.ndarray) else y_hat
