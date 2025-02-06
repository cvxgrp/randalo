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
class Regularizer(ABC):
    linear: np.ndarray | list[int] | slice = field(default=None)
    scale: float = field(init=False, default=1.0)
    parameter: HyperParameter = field(init=False, default=None)

    def __mul__(self, r):
        if isinstance(r, HyperParameter):
            if self.parameter is not None:
                raise TypeError("Cannot have multiple parameters")
            out = type(self)(linear=self.linear)
            out.scale = self.scale
            out.parameter= r
            return out
        # elif isinstance(r, float | np.float32): # <- didn't allow integers or other floats...
        else:
            out = type(self)(linear=self.linear)
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
    def get_constraint_hessian_mask(self, beta_hat, epsilon=1e-6):
        """
        Returns two matrices and a vector[bool] (or None if they would be the
        0 matrix or all true) such that
        the hessian of this regularizer can be viewed as 
            infinity * (first matrix) + (second matrix) + infty * (third matrix)
        first matrix is called the "constraint matrix" second matrix is called
        the "hessian matrix"

        The third matrix is given by
            diag = torch.zero(n)
            diag[mask] = 1
            third matrix = torch.diag(diag)
        ie any entry associated with a False in mask is held to always be zeros
        """

    def get_constraint_hessian_mask_sparse(self, beta_hat, mask, p, epsilon=1e-6):
        beta = torch.zeros(p, device=beta_hat.device, dtype=beta_hat.dtype)
        beta[mask] = beta_hat
        return self.get_constraint_hessian_mask(beta, epsilon)

    def _scale(self):
        if self.parameter is not None:
            return self.scale * self.parameter.value
        else:
            return self.scale


class SquareRegularizer(Regularizer):
    def to_cvxpy(self, variable):
        super().to_cvxpy(variable, cp.sum_squares)

    def get_constraint_hessian_mask(self, beta_hat, epsilon=1e-6):
        return self._internal(beta_hat.shape, beta_hat.dtype, beta_hat.device, epsilon)

    def get_constraint_hessian_mask_sparse(self, beta_hat, mask, p, epsilon=1e-6):
        return self._internal(mask.shape, beta_hat.dtype, beta_hat.device, epsilon)

    def _internal(self, shape, dtype, device, epsilon):
        scale = 2 * self._scale()
        if scale == 0.0:
            return None, None, None
        if self.linear is None:
            return None, torch.diag(
                    scale * torch.ones(shape, dtype=dtype, device=device)), None
        elif isinstance(self.linear, (list, slice)):
            diag = torch.zeros(shape, dtype=dtype, device=device)
            diag[self.linear] = scale
            return None, torch.diag(diag), None
        else:
            A = utils.to_tensor(self.linear)
            return None, torch.diag(scale * (A.mT @ A)), None


class L1Regularizer(Regularizer):
    def to_cvxpy(self, variable):
        super().to_cvxpy(variable, cp.norm1)

    def get_constraint_hessian_mask(self, beta_hat, epsilon=1e-6):
        mask = torch.ones_like(beta_hat, dtype=bool)
        if self.linear is None:
            mask[torch.abs(beta_hat) <= epsilon] = False
            return None, None, mask
        elif isinstance(self.linear, (list, slice)):
            mask[self.linear][torch.abs(beta_hat[self.linear]) <= epsilon] = False
            return None, None, mask
        else:
            A = utils.from_numpy(self.linear)
            return A[torch.abs(A @ beta_hat) <= epsilon, :], None, None

    def get_constraint_hessian_mask_sparse(self, beta_hat, mask, p, epsilon=1e-6):
        mask_0 = torch.ones_like(beta_hat, dtype=bool)
        if self.linear is None:
            mask_0[torch.abs(beta_hat) <= epsilon] = False
            return None, None, mask_0
        elif isinstance(self.linear, slice):
            if mask.dtype == bool or mask.dtype == torch.bool:
                raise NotImplementedError()
                mask_0[mask[self.linear]][torch.abs(beta_hat[idx]) <= epsilon] = False
            else:
                start, end, step = self.linear.indices(p)
                mask_1 = (mask >= start) & (mask  < end) & ((mask - start) % step == 0)
                mask_0[mask_1][torch.abs(beta_hat[mask_1]) <= epsilon] = False
            return None, None, mask_0
        elif isinstance(self.linear, (list, slice)):
            idx = torch.cumsum(mask_0)[self.linear]
            mask_0[idx][torch.abs(beta_hat[idx]) <= epsilon] = False
            return None, None, mask_0
        else:
            return super().get_constraint_hessian_mask_sparse(beta_hat, mask, p, epsilon)


class L2Regularizer(Regularizer):
    def to_cvxpy(self, variable):
        super().to_cvxpy(variable, cp.norm2)

    def get_constraint_hessian_mask(self, beta_hat, epsilon=1e-6):
        linear = self.linear
        if linear is None:
            norm = torch.linalg.norm(beta_hat)
            if norm <= epsilon:
                mask = torch.zeros_like(beta_hat, dtype=bool)
                return None, None, mask

            tilde_beta_hat_2d = torch.atleast_2d(beta_hat).T / norm
            hessian = self._scale() * (torch.eye(beta_hat.shape) - beta_hat_2d @ beta_hat_2d.T)
            return None, hessian, None
        elif isinstance(linear, (list, slice)):
            norm = torch.linalg.norm(beta_hat[linear])
            if norm <= epsilon:
                mask = torch.ones_like(beta_hat, dtype=bool)
                mask[linear] = False
                return None, None, mask
            tilde_b = torch.atleast_2d(torch.zeros_like(beta_hat)).T / norm
            tilde_b[linear] = beta_hat[linear]
            diag = torch.zero_like(beta_hat)
            diag[linear] = 1.0
            return None, self._scale() * (torch.diag(diag) - tilde_b @ tilde_b.T), None

        else:
            Lb = linear @ beta_hat
            norm = torch.linalg.norm(Lb)
            tilde_Lb = torch.atleast_2d(Lb).T / norm
            if Lb <= epsilon:
                return linear, None, None
            return None, self._scale() * linear.T @ (
                    torch.eye(Lb.shape[0]) - Lb_2d @ Lb_2d.T) @ linear, None


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

    def get_constraint_hessian_mask(self, beta_hat, epsilon=1e-6):
        constraints = []
        hessians = []
        mask = torch.ones_like(beta_hat, dtype=bool)
        for reg in self.exprs:
            cons, hess, m = reg.get_constraint_hessian_mask(beta_hat, epsilon)
            if cons is not None:
                constraints.append(cons)
            if hess is not None:
                hessians.append(hess)
            if m is not None:
                mask &= m

        constraints = torch.vstack(constraints) if len(constraints) > 0 else None
        hessians = sum(hessians) if len(hessians) > 0 else None
        return constraints, hessians, mask
 
    def get_constraint_hessian_mask_sparse(self, beta_hat, mask_1, p, epsilon=1e-6):
        constraints = []
        hessians = []
        mask = torch.ones_like(beta_hat, dtype=bool)
        for reg in self.exprs:
            cons, hess, m = reg.get_constraint_hessian_mask_sparse(beta_hat, mask_1, p, epsilon)
            if cons is not None:
                constraints.append(cons)
            if hess is not None:
                hessians.append(hess)
            if m is not None:
                mask &= m

        constraints = torch.vstack(constraints) if len(constraints) > 0 else None
        hessians = sum(hessians) if len(hessians) > 0 else None
        return constraints, hessians, mask


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
    def __init__(self, weights=None):
        self.weights = weights
    def func(self, y, z):
        return self.weights * (y - z) ** 2

    def to_cvxpy(self, y, z):
        if self.weights is not None:
            return cp.sum(cp.multiply(weights, cp.square(y - z)))
        else:
            return cp.sum_squares(y - z) / np.prod(y.shape)
