from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import torch


@dataclass
class HyperParameter:
    parameter: cp.Parameter = field(default_factory=cp.Parameter)
    scale: float = field(init=False, default=1.0)

    def __mul__(self, r):
        # TODO: __mul__ should not have side-effects
        if isinstance(r, float | np.float32):
            self.scale *= r
        else:
            return NotImplemented
        return self

    @property
    def value(self):
        return self.scale * self.parameter.value

    @value.setter
    def value(self, val):
        assert self.scale == 1.0, "Cannot set the value of a scaled parameter"
        self.parameter.value = val


@dataclass
class Regularizer:
    linear: np.ndarray | list[int] = field(default=None)
    scale: float = field(init=False, default=1.0)
    parameter: HyperParameter = field(init=False, default=None)

    def __mul__(self, r):
        # TODO: __mul__ should not have side-effects
        if isinstance(r, HyperParameter):
            if self.parameter is not None:
                raise TypeError("Cannot have multiple parameters")
            self.parameter = r
        elif isinstance(r, float | np.float32):
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

    def __mul__(self, r):
        return Sum([r * expr for expr in self.exprs])

    def __rmul__(self, r):
        return self * r

    def __add__(self, r):
        if isinstance(r, Sum):
            return Sum(r.exprs + self.exprs)
        else:
            return NotImplemented


# @dataclass
class Loss:
    # we don't need to store data with the loss
    # y: torch.Tensor
    # X: torch.Tensor
    pass


class LogisticLoss(Loss):
    @staticmethod
    def func(y, y_hat):
        return torch.log(1 + torch.exp(-y * y_hat))


class MSELoss(Loss):
    def __call__(self, y, z):
        return torch.mean((y - z) ** 2)

    @staticmethod
    def func(y, y_hat):
        return (y - y_hat) ** 2 / 2 / np.prod(y.shape)
