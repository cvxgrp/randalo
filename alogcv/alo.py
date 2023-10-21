from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

import torch
from torch import autograd, Tensor

import linops as lo
from linops import LinearOperator

from . import utils
from .truncnorm import truncnorm_mean


class ALOBase(ABC):
    """Base class for ALO estimators."""

    def __init__(
        self, loss_fun: Callable[[Tensor, Tensor], Tensor], y: Tensor, y_hat: Tensor
    ):
        """Initialize an ALO estimator.

        Parameters
        ----------
        loss_fun : Callable[[Tensor, Tensor], Tensor]
            The loss function. Accepts two n-dimensional tensors of the true response and
            prediction and returns an n-dimensional tensor of the loss values.
        y : Tensor
            The true responses.
        y_hat : Tensor
            The predictions.
        """

        self._loss_fun = loss_fun

        # detach and clone to avoid memory leaks
        self._y = y.detach().clone().requires_grad_(True)
        self._y_hat = y_hat.detach().clone().requires_grad_(True)
        self.n = self._y.shape[0]

        self._device = self._y.device

        # compute first and second derivatives of loss function
        # we obtain the vector derivatives by summing and then taking the gradient
        loss = loss_fun(self._y, self._y_hat).sum()
        # keep the graph for computing the second derivatives
        self._dloss_dy_hat, *_ = autograd.grad(loss, self._y_hat, create_graph=True)
        dloss_dy_hat_sum = self._dloss_dy_hat.sum()

        self._d2loss_dboth, self._d2loss_dy_hat2 = autograd.grad(
            dloss_dy_hat_sum, [self._y, self._y_hat], allow_unused=True
        )
        if self._d2loss_dboth is None:
            self._d2loss_dboth = torch.zeros_like(self._y)
        if self._d2loss_dy_hat2 is None:
            self._d2loss_dy_hat2 = torch.zeros_like(self._y_hat)

        # free GPU memory used by autograd
        self._y = self._y.detach()
        self._y_hat = self._y_hat.detach()
        self._dloss_dy_hat = self._dloss_dy_hat.detach()
        self._d2loss_dboth = self._d2loss_dboth.detach()
        self._d2loss_dy_hat2 = self._d2loss_dy_hat2.detach()

        # precompute some quantities for working with generic form
        # a = dloss_dy_hat
        # b = d2loss_dy_hat2
        # c = d2loss_dboth
        self.a_c = self._dloss_dy_hat / self._d2loss_dy_hat2
        self.c_b = self._d2loss_dy_hat2 / self._d2loss_dboth

    def transform_jac(self, diag_jac: Tensor) -> Tensor:
        """Transform the Jacobian into a generic form.

        For general losses, the Jacobian may not cover the range from 0 to 1.
        This function transforms the Jacobian into a generic form that does
        cover the range from 0 to 1 by eliminating loss-specific terms.

        Parameters
        ----------
        diag_jac : Tensor
            The diagonal of the Jacobian. If a multi-dimensional tensor, the last
            dimension should be the same length as y and y_hat.

        Returns
        -------
        Tensor
            The transformed diagonal of the Jacobian.
        """
        return -diag_jac * self.c_b.expand_as(diag_jac)

    def y_tilde_from_transformed_jac(self, transformed_jac: Tensor) -> Tensor:
        """Compute the corrected predictions using the transformed Jacobian.

        Parameters
        ----------
        transformed_jac : Tensor
            The transformed diagonal of the Jacobian.

        Returns
        -------
        Tensor
            The corrected predictions.
        """
        transformed_jac = torch.clamp(transformed_jac, 0, 1)
        return self._y_hat + self.a_c * transformed_jac / (1 - transformed_jac)

    def y_tilde_from_jac(self, diag_jac: Tensor) -> Tensor:
        """Convenience method for computing the corrected predictions.

        First transforms the diagonal of the Jacobian into a generic form,
        then computes the corrected predictions.

        Parameters
        ----------
        diag_jac : Tensor
            The diagonal of the Jacobian.

        Returns
        -------
        Tensor
            The corrected predictions.
        """

        return self.y_tilde_from_transformed_jac(self.transform_jac(diag_jac))

    @abstractmethod
    def joint_vars(self) -> [Tensor, Tensor]:
        """Return the true response and corrected predictions."""
        pass

    @abstractmethod
    def eval_risk(
        self, risk: Callable[[Tensor, Tensor], Tensor], *args, **kwargs
    ) -> float:
        """Evaluate a risk function.

        Parameters
        ----------
        risk : Callable[[Tensor, Tensor], Tensor]
            The risk function. Accepts two n-dimensional tensors of the true response and
            corrected predictions and returns an n-dimensional tensor of the risk values.

        Returns
        -------
        float
            The risk estimate.
        """
        pass


class ALOExact(ALOBase):
    """Exact ALO estimator."""

    def __init__(
        self,
        loss_fun: Callable[[Tensor, Tensor], Tensor],
        y: Tensor,
        y_hat: Tensor,
        diag_jac: Tensor,
    ):
        """Initialize an exact ALO estimator.

        Parameters
        ----------
        loss_fun : Callable[[Tensor, Tensor], Tensor]
            The loss function.
        y : Tensor
            The true response.
        y_hat : Tensor
            The predictions.
        """
        super().__init__(loss_fun, y, y_hat)
        self._diag_jac = diag_jac
        self._y_tilde = self.y_tilde_from_jac(self._diag_jac)

    def joint_vars(self) -> [Tensor, Tensor]:
        """Return the true response and corrected predictions."""
        return self._y, self._y_tilde

    def eval_risk(self, risk: Callable[[Tensor, Tensor], Tensor]) -> float:
        """Evaluate a risk function.

        Parameters
        ----------
        risk : Callable[[Tensor, Tensor], Tensor]
            The risk function.

        Returns
        -------
        float
            The risk estimate.
        """
        return risk(self._y, self._y_tilde).sum().item()


class ALORandomized(ALOBase):
    """Randomized ALO estimator."""

    def __init__(
        self,
        loss_fun: Callable[[Tensor, Tensor], Tensor],
        y: Tensor,
        y_hat: Tensor,
        jac: LinearOperator,
        m: int,
        generator: Optional[torch.Generator] = None,
    ):
        """Initialize a randomized ALO estimator.

        Parameters
        ----------
        loss_fun : Callable[[Tensor, Tensor], Tensor]
            The loss function.
        y : Tensor
            The true response.
        y_hat : Tensor
            The predictions.
        jac : LinearOperator
            The Jacobian.
        m : int
            The number of samples to initialize with.
        generator : Optional[torch.Generator]
            The random number generator.
        """
        super().__init__(loss_fun, y, y_hat)
        self._jac = jac

        self._transformed_diag_jac_estims = None
        self.m = 0

        if generator is None:
            generator = torch.Generator(device=self._device)
        self._generator = generator

        self._best_transformed_diag_jac = None
        self._transformed_diag_jac_mean = None
        self._transformed_diag_jac_std = None
        self.do_diag_jac_estims_upto(m)

    def _get_matvecs(self, m: int) -> [Tensor, Tensor]:
        """Compute random matrix-vector products of the Jacobian using random Rademacher vectors.

        Parameters
        ----------
        m : int
            The number of samples to compute.

        Returns
        -------
        [Tensor, Tensor]
            The matrix-vector products and the random vectors.
        """
        Omega = torch.randint(0, 2, (self.n, m), generator=self._generator) * 2.0 - 1
        return self._jac @ Omega, Omega

    def do_diag_jac_estims_upto(self, m: int) -> None:
        """Compute more diagonal Jacobian estimates.

        Parameters
        ----------
        m : int
            The desired total number of samples.
        """

        if m <= self.m:
            raise ValueError("m must be greater than the current number of samples")

        # compute the matrix-vector products
        matvecs, Omega = self._get_matvecs(m - self.m)

        # update the diagonal Jacobian estimates
        transformed_diag_jac_estims = self.transform_jac((matvecs * Omega).T).T
        if self._transformed_diag_jac_estims is None:
            self._transformed_diag_jac_estims = transformed_diag_jac_estims
        else:
            self._transformed_diag_jac_estims = torch.cat(
                (self._transformed_diag_jac_estims, transformed_diag_jac_estims), dim=1
            )
        self.m = m

        # compute the sufficient statistics and construct the truncated normal estimate
        self._transformed_diag_jac_mean = self._transformed_diag_jac_estims.mean(dim=1)
        self._transformed_diag_jac_std = self._transformed_diag_jac_estims.std(dim=1)
        self._best_transformed_diag_jac = truncnorm_mean(
            self._transformed_diag_jac_mean,
            self._transformed_diag_jac_std / np.sqrt(m),
            torch.tensor([0], device=self._device),
            torch.tensor([1], device=self._device),
        )

    def joint_vars(self) -> [Tensor, Tensor]:
        """Return the true response and corrected predictions."""
        return self._y, self.y_tilde_from_transformed_jac(
            self._best_transformed_diag_jac
        )

    def eval_risk(
        self,
        risk: Callable[[Tensor, Tensor], Tensor],
        order: Optional[int] = 1,
        power: float = 1.0,
        n_points: int = 50,
    ) -> float:
        """Evaluate a risk function.

        Parameters
        ----------
        risk : Callable[[Tensor, Tensor], Tensor]
            The risk function.
        order : Optional[int]
            The order of the polynomial to fit to the risk estimates. If None, return the
            risk estimate for the best diagonal Jacobian estimate.
        power : float
            The power of the polynomial to fit to the risk estimates.
        n_points : int
            The number of points to use for the polynomial fitting.

        Returns
        -------
        float
            The risk estimate.
        """

        # if order is not provided, return the risk estimate for the best diagonal Jacobian
        if order is None:
            return risk(*self.joint_vars()).sum().item()

        # otherwise, estimate the risk through polynomial fitting
        if self.m <= 1:
            raise ValueError("m must be greater than 1")
        m0 = self.m // 2

        # initialize the arrays for the polynomial fitting
        ms = np.linspace(m0, self.m, n_points).astype(int)
        risks = np.zeros(n_points)

        # iterate over the number of samples
        for i, m in enumerate(ms):
            # estimate the diagonal Jacobian with m samples
            subset_sketched = self._transformed_diag_jac_estims[
                :,
                torch.randperm(self.m, generator=self._generator)[:m],
            ]
            diag_jac_mean = subset_sketched.mean(dim=1)
            diag_jac_std = self._transformed_diag_jac_std / np.sqrt(m)

            # estimate via truncated normal
            diag_jac = truncnorm_mean(
                diag_jac_mean,
                diag_jac_std,
                torch.tensor([0], device=self._device),
                torch.tensor([1], device=self._device),
            )

            # compute the risk estimate
            risks[i] = (
                risk(self._y, self.y_tilde_from_transformed_jac(diag_jac)).sum().item()
            )

        # return the constant term of the polynomial fit
        self._ms = ms
        self._risks = risks
        coefs, self._res_m_to_risk_fit = utils.robust_poly_fit(
            1 / ms**power, risks, order
        )
        return coefs[0]
