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
            return HyperParameter(parameter=self.parameter, scale=self.scale * r)
        else:
            return NotImplemented

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
        if isinstance(r, HyperParameter):
            if self.parameter is not None:
                raise TypeError("Cannot have multiple parameters")
            out = Regularizer(linear=self.linear)
            out.scale = self.scale
            out.parameter= r
            return out
        # elif isinstance(r, float | np.float32): # <- didn't allow integers or other floats...
        else:
            out = Regularizer(linear=self.linear)
            out.scale = self.scale * r
            out.parameter = self.parameter
            return out
        # else:
        #     raise TypeError("Multiply must be with either a scalar or HyperParameter")
        return NotImplemented

    def __rmul__(self, r):
        return self * r

    def __truediv__(self, r):
        return self * (1 / r)

    def __add__(self, rhs):
        if rhs == 0:
            return self
        if isinstance(rhs, Sum):
            return Sum([self] + rhs.exprs)
        elif isinstance(rhs, Regularizer):
            return Sum([self, rhs])
        else:
            raise TypeError("Addend must be a Regularizer or Sum")

    def __radd__(self, lhs):
        return self + lhs

    @abstractmethod
    def to_cvxpy(self, variable, func):
        expr = func(obj.linear @ variable if obj.linear is not None else variable)
        if obj.parameter is None:
            return obj.scale * expr
        else:
            return obj.scale * obj.parameter.scale * obj.parameter.parameter * expr

    @abstractmethod
    def get_constraint_hessian(self, mask, beta_hat, epsilon=1e-6):
        """
        Returns two matrices (or None if they would be the 0 matrix) such that
        the hessian of this regularizer can be viewed as 
            infinity * (first matrix) + (second matrix)
        first matrix is called the "constraint matrix" second matrix is called
        the "hessian matrix"

        If some rows of the constraint matrix would be rows of the identity
        matrix this function can instead modify mask to contain False in
        the index corresponding to which row of the identity it is
        """

    def _scale(self):
        if regularizer.parameter is not None:
            scale = regularizer.scale * regularizer.parameter.value
        else:
            scale = regularizer.scale


class SquareRegularizer(Regularizer):
    def to_cvxpy(self, variable):
        super().to_cvxpy(variable, cp.sum_squares)

    def get_constraint_hessian(self, mask, beta_hat, epsilon=1e-6):
        scale = self._scale()
        if linear is None:
            return None, torch.diag(
                    2 * scale * torch.ones_like(mask, dtype=beta_hat.dtype))
        elif isinstance(linear, list):
            diag = torch.zeros_like(mask, dtype=beta_hat.dtype)
            diag[linear] = scale
            return None, torch.diag(diag)
        else:
            A = utils.to_tensor(linear)
            return None, torch.diag(scale * (A.mT @ A))


class L1Regularizer(Regularizer):
    def to_cvxpy(self, variable):
        super().to_cvxpy(variable, cp.norm1)

    def get_constraint_hessian(self, mask, beta_hat, epsilon=1e-6):
        scale = self._scale()
        if linear is None:
            mask[torch.abs(beta_hat) <= epsilon] = False
            return None, None
        elif isinstance(linear, list):
            mask[linear][torch.abs(beta_hat[linear]) <= epsilon] = False
            return None, None
        else:
            A = utils.from_numpy(linear)
            return A[torch.abs(A @ beta_hat) <= epsilon, :], None


class L2Regularizer(Regularizer):
    def to_cvxpy(self, variable):
        super().to_cvxpy(variable, cp.norm2)

    def get_constraint_hessian(self, mask, beta_hat, epsilon=1e-6):
        linear = self.linear
        if linear is None:
            norm = torch.linalg.norm(beta_hat)
            if norm <= epsilon:
                mask[:] = False
                return None, None

            tilde_beta_hat_2d = torch.atleast_2d(beta_hat).T / norm
            return None, self._scale() * (torch.eye(beta_hat.shape) - beta_hat_2d @ beta_hat_2d.T)
        elif isinstance(linear, list):
            norm = torch.linalg.norm(beta_hat[linear])
            if norm <= epsilon:
                mask[linear] = False
                return None, None
            tilde_b = torch.atleast_2d(torch.zeros_like(beta_hat)).T / norm
            tilde_b[linear] = beta_hat[linear]
            diag = torch.zero_like(beta_hat)
            diag[linear] = 1.0
            return None, self._scale() * (torch.diag(diag) - tilde_b @ tilde_b.T)
        else:
            Lb = linear @ beta_hat
            norm = torch.linalg.norm(Lb)
            tilde_Lb = torch.atleast_2d(Lb).T / norm
            if Lb <= epsilon:
                return linear, None
            return None, self._scale() * linear.T @ (
                    torch.eye(Lb.shape[0]) - Lb_2d @ Lb_2d.T) @ linear


class HuberRegularizer(Regularizer):
    def to_cvxpy(self, variable):
        super().to_cvxpy(variable, cp.huber)


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

    def to_cvxpy(self, variable):
        return cp.sum([term.to_cvxpy(variable) for term in terms])

    def get_constraint_hessian(self, mask, beta_hat, epsilon=1e-6):
        constraints = []
        hessians = []
        for reg in exprs:
            cons, hess = reg.get_constraint_hessian(mask, beta_hat, epsilon)
            if cons is not None:
                constraints.append(cons)
            if hess is not None:
                hessians.append(hess)

        constraints = torch.vstack(constraints) if len(constraints) > 0 else None
        hessians = sum(hessians) if len(hessians) > 0 else None
        return constraints, hessians
 

class Loss(ABC):
    def __call__(self, y, z):
        return torch.mean(self.func(y, z))

    @abstractmethod
    def func(self, y, z):
        pass

    @abstractmethod
    def to_cvxpy(self, X, y, variable):
        pass


class LogisticLoss(Loss):
    def func(self, y, z):
        return torch.log(1 + torch.exp(-y * z))

    def to_cvxpy(self, y, z):
        return cp.sum(cp.logistic(-cp.multiply(y, z))) / np.prod(y.shape)
                 

class MSELoss(Loss):
    def func(self, y, z):
        return (y - z) ** 2

    def to_cvxpy(self, y, z):
        return cp.sum_squares(y - z) / np.prod(y.shape)
