from dataclasses import dataclass, field

import cvxpy as cp
import torch
import numpy as np


class HyperParameter:
    parameter = field(default_facultory=cp.Paramater)


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


@dataclass
class Loss:
    y: torch.Tensor
    X: torch.Tensor


class LogisticLoss(Loss):
    pass


class MSELoss(Loss):
    pass
