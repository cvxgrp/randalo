from abc import ABC, abstractmethod
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
        # elif isinstance(r, float | np.float32): # <- didn't allow integers or other floats...
        else:
            self.scale *= r
        # else:
        #     raise TypeError("Multiply must be with either a scalar or HyperParameter")
        return self

    def __rmul__(self, r):
        return self * r

    def __truediv__(self, r):
        return self * (1 / r)

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

    def __truediv__(self, r):
        return self * (1 / r)

    def __add__(self, r):
        if isinstance(r, Sum):
            return Sum(r.exprs + self.exprs)
        else:
            return NotImplemented


class Loss:
    def __call__(self, y, z):
        return torch.mean(self.func(y, z))

    @abstractmethod
    def func(self, y, z):
        pass


class LogisticLoss(Loss):
    def func(self, y, z):
        return torch.log(1 + torch.exp(-y * z))


class MSELoss(Loss):
    def func(self, y, z):
        return (y - z) ** 2
