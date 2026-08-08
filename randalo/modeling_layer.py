from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib

import numpy as np
import torch


def _require_cvxpy():
    try:
        return importlib.import_module("cvxpy")
    except ModuleNotFoundError as error:
        if error.name != "cvxpy":
            raise
        raise ModuleNotFoundError(
            "CVXPY support requires the optional dependency; "
            "install it with `pip install 'randalo[cvxpy]'`."
        ) from error


class _CVXPYProxy:
    def __getattr__(self, name):
        return getattr(_require_cvxpy(), name)


cp = _CVXPYProxy()


@dataclass
class HyperParameter:
    parameter: object = field(default_factory=lambda: cp.Parameter())
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
    linear: np.ndarray | list[int] = field(default=None)
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
        if self.linear is None:
            argument = variable
        elif isinstance(self.linear, list):
            argument = variable[self.linear]
        else:
            argument = self.linear @ variable

        expression = func(argument)
        if self.parameter is None:
            return self.scale * expression
        return (
            self.scale
            * self.parameter.scale
            * self.parameter.parameter
            * expression
        )

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

    def _scale(self):
        if self.parameter is not None:
            return self.scale * self.parameter.value
        else:
            return self.scale


class SquareRegularizer(Regularizer):
    def to_cvxpy(self, variable):
        return super().to_cvxpy(variable, cp.sum_squares)

    def get_constraint_hessian_mask(self, beta_hat, epsilon=1e-6):
        scale = self._scale()
        if scale == 0.0:
            return None, None, None
        if self.linear is None:
            return None, torch.diag(
                    2 * scale * torch.ones_like(beta_hat, dtype=beta_hat.dtype)), None
        elif isinstance(self.linear, list):
            diag = torch.zeros_like(beta_hat)
            diag[self.linear] = 2 * scale
            return None, torch.diag(diag), None
        else:
            A = torch.as_tensor(
                self.linear, dtype=beta_hat.dtype, device=beta_hat.device
            )
            return None, 2 * scale * (A.mT @ A), None


class L1Regularizer(Regularizer):
    def to_cvxpy(self, variable):
        return super().to_cvxpy(variable, cp.norm1)

    def get_constraint_hessian_mask(self, beta_hat, epsilon=1e-6):
        mask = torch.ones_like(beta_hat, dtype=bool)
        if self.linear is None:
            mask[torch.abs(beta_hat) <= epsilon] = False
            return None, None, mask
        elif isinstance(self.linear, list):
            indices = torch.as_tensor(
                self.linear, dtype=torch.long, device=beta_hat.device
            )
            mask[indices] = torch.abs(beta_hat[indices]) > epsilon
            return None, None, mask
        else:
            A = torch.as_tensor(
                self.linear, dtype=beta_hat.dtype, device=beta_hat.device
            )
            return A[torch.abs(A @ beta_hat) <= epsilon, :], None, None


class ZeroRegularizer(Regularizer):
    """A regularizer with no value, curvature, or active constraints."""

    def to_cvxpy(self, variable):
        return cp.Constant(0.0)

    def get_constraint_hessian_mask(self, beta_hat, epsilon=1e-6):
        return None, None, None


class NonNegativeRegularizer(Regularizer):
    """The indicator of nonnegative coefficients.

    ``linear`` may select the coefficients constrained to be nonnegative. This
    is primarily used to reproduce scikit-learn's ``positive=True`` active set.
    """

    def to_cvxpy(self, variable):
        constrained = variable if self.linear is None else variable[self.linear]
        return cp.transforms.indicator([constrained >= 0])

    def get_constraint_hessian_mask(self, beta_hat, epsilon=1e-6):
        mask = torch.ones_like(beta_hat, dtype=bool)
        if self.linear is None:
            mask = beta_hat > epsilon
        else:
            indices = torch.as_tensor(
                self.linear, dtype=torch.long, device=beta_hat.device
            )
            mask[indices] = beta_hat[indices] > epsilon
        return None, None, mask


class L2Regularizer(Regularizer):
    def to_cvxpy(self, variable):
        return super().to_cvxpy(variable, cp.norm2)

    def get_constraint_hessian_mask(self, beta_hat, epsilon=1e-6):
        linear = self.linear
        if self.linear is None:
            norm = torch.linalg.norm(beta_hat)
            if norm <= epsilon:
                mask = torch.zeros_like(beta_hat, dtype=bool)
                return None, None, mask

            unit_beta = beta_hat / norm
            identity = torch.eye(
                beta_hat.numel(), dtype=beta_hat.dtype, device=beta_hat.device
            )
            hessian = self._scale() / norm * (
                identity - torch.outer(unit_beta, unit_beta)
            )
            return None, hessian, None
        elif isinstance(linear, list):
            indices = torch.as_tensor(
                linear, dtype=torch.long, device=beta_hat.device
            )
            norm = torch.linalg.norm(beta_hat[indices])
            if norm <= epsilon:
                mask = torch.ones_like(beta_hat, dtype=bool)
                mask[indices] = False
                return None, None, mask
            unit_beta = torch.zeros_like(beta_hat)
            unit_beta[indices] = beta_hat[indices] / norm
            diagonal = torch.zeros_like(beta_hat)
            diagonal[indices] = 1.0
            hessian = self._scale() / norm * (
                torch.diag(diagonal) - torch.outer(unit_beta, unit_beta)
            )
            return None, hessian, None

        else:
            linear = torch.as_tensor(
                linear, dtype=beta_hat.dtype, device=beta_hat.device
            )
            Lb = linear @ beta_hat
            norm = torch.linalg.norm(Lb)
            if norm <= epsilon:
                return linear, None, None
            unit_Lb = Lb / norm
            identity = torch.eye(
                Lb.numel(), dtype=Lb.dtype, device=Lb.device
            )
            hessian = self._scale() / norm * linear.T @ (
                identity - torch.outer(unit_Lb, unit_Lb)
            ) @ linear
            return None, hessian, None


class HuberRegularizer(Regularizer):
    def to_cvxpy(self, variable):
        return super().to_cvxpy(variable, lambda value: cp.sum(cp.huber(value)))

    def get_constraint_hessian_mask(self, beta_hat, epsilon=1e-6):
        scale = self._scale()
        if self.linear is None:
            diagonal = 2 * scale * (torch.abs(beta_hat) < 1 - epsilon)
            return None, torch.diag(diagonal.to(beta_hat.dtype)), None
        if isinstance(self.linear, list):
            indices = torch.as_tensor(
                self.linear, dtype=torch.long, device=beta_hat.device
            )
            diagonal = torch.zeros_like(beta_hat)
            quadratic = torch.abs(beta_hat[indices]) < 1 - epsilon
            diagonal[indices] = 2 * scale * quadratic.to(beta_hat.dtype)
            return None, torch.diag(diagonal), None

        linear = torch.as_tensor(
            self.linear, dtype=beta_hat.dtype, device=beta_hat.device
        )
        quadratic = torch.abs(linear @ beta_hat) < 1 - epsilon
        active_linear = linear[quadratic]
        return None, 2 * scale * active_linear.T @ active_linear, None


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
        return cp.sum([term.to_cvxpy(variable) for term in self.exprs])

    def get_constraint_hessian_mask(self, beta_hat, epsilon=1e-6):
        constraints = []
        hessians = []
        mask = torch.ones_like(beta_hat, dtype=bool)
        has_mask = False
        for reg in self.exprs:
            cons, hess, m = reg.get_constraint_hessian_mask(beta_hat, epsilon)
            if cons is not None:
                constraints.append(cons)
            if hess is not None:
                hessians.append(hess)
            if m is not None:
                mask &= m
                has_mask = True

        constraints = torch.vstack(constraints) if len(constraints) > 0 else None
        hessians = sum(hessians) if len(hessians) > 0 else None
        return constraints, hessians, mask if has_mask else None
 

class Loss(ABC):
    def __call__(self, y, z):
        return torch.mean(self.func(y, z))

    @abstractmethod
    def func(self, y, z):
        pass

    @abstractmethod
    def to_cvxpy(self, X, y, variable):
        pass


class WeightedLoss(Loss):
    """Apply fixed per-sample weights to an elementwise loss.

    The weights are expected to have mean one, so this class preserves the
    objective scale of the wrapped loss.
    """

    def __init__(self, loss, sample_weight):
        self.loss = loss
        self.sample_weight = torch.as_tensor(sample_weight).detach().clone()

    def func(self, y, z):
        sample_weight = self.sample_weight.to(dtype=z.dtype, device=z.device)
        return sample_weight * self.loss.func(y, z)

    def to_cvxpy(self, y, z):
        sample_weight = np.asarray(self.sample_weight)
        if isinstance(self.loss, MSELoss):
            elementwise_loss = cp.square(y - z)
        elif isinstance(self.loss, LogisticLoss):
            elementwise_loss = cp.logistic(-cp.multiply(y, z))
        else:
            raise NotImplementedError(
                "CVXPY conversion is not implemented for this weighted loss."
            )
        return cp.sum(cp.multiply(sample_weight, elementwise_loss)) / np.prod(y.shape)


class LogisticLoss(Loss):
    def func(self, y, z):
        return torch.nn.functional.softplus(-y * z)

    def to_cvxpy(self, y, z):
        return cp.sum(cp.logistic(-cp.multiply(y, z))) / np.prod(y.shape)
                 

class MSELoss(Loss):
    def func(self, y, z):
        return (y - z) ** 2

    def to_cvxpy(self, y, z):
        return cp.sum_squares(y - z) / np.prod(y.shape)
