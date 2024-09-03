from dataclasses import dataclass, field

import cvxpy as cp
import torch
import numpy as np


class HyperParameter:
    parameter = field(default_factory=cp.Parameter)


@dataclass
class Regularizer:
    linear: np.ndarray = field(default=None)
    scale: float = field(init=False, default=1.0)
    parameter: HyperParameter = field(init=False, default=None)

    def __mul__(self, r):
        if isinstance(r, HyperParameter):
            assert self.parameter is None
            parameter = r
        elif isinstance(r, float | np.float32 | torch.float64 | torch.float32):
            self.scale *= r
        else:
            raise TypeError("Multiply must be with either a scalar or HyperParameter")
        return self

    def __rmul__(self, r):
        return self * r

    def __add__(self, rhs):
        if isinstance(rhs, Sum):
            return Sum([self] + rhs.exprs)
        elif isinstance(rhs, Regularizer):
            return Sum([self, rhs])
        else:
            raise TypeError("Addend must be a Regularizer or Sum")

    def __radd__(self, lhs):
        return self + lhs


class SquareRegularizer(Regularizer):
    pass


class L1Regularizer(Regularizer):
    pass


class L2Regularizer(Regularizer):
    pass


class HuberRegularizer(Regularizer):
    pass


@dataclass
class Sum:
    exprs: list[Regularizer]


# @dataclass
class Loss:
    # we don't need to store data with the loss
    # y: torch.Tensor
    # X: torch.Tensor
    pass


class LogisticLoss(Loss):
    pass


class MSELoss(Loss):

    def __call__(self, y, z):
        return (y - z) ** 2 / 2
