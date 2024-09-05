from typing import Callable

import linops as lo
import numpy as np
import sklearn.utils.validation
import torch

from . import modeling_layer as ml
from . import truncnorm
from . import utils


class RandALO(object):
    def __init__(
        self,
        loss: ml.Loss = None,
        jac: lo.LinearOperator = None,
        y: torch.Tensor | np.ndarray = None,
        y_hat: torch.Tensor | np.ndarray = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        rng: torch.Generator = None,
    ):
        """Initialize the RandALO object.

        Parameters
        ----------
        loss : Loss
            Loss function.
        jac : LinearOperator
            Jacobian operator.
        y : torch.Tensor | np.ndarray
            True labels.
        y_hat : torch.Tensor | np.ndarray
            Predicted values.
        dtype : torch.dtype, optional
            Data type for tensors, by default torch.float32.
        device : torch.device, optional
            Device for tensors, by default None.
        rng : torch.Generator, optional
            Random number generator, by default None.
        """

        if loss is None:
            raise ValueError("loss function must be provided")
        self._loss = loss

        if jac is None:
            raise ValueError("Jacobian operator must be provided")
        self._jac = jac

        if y is None:
            raise ValueError("label values must be provided")
        y = utils.to_tensor(y, dtype=dtype, device=device)

        if y_hat is None:
            raise ValueError("predicted values must be provided")
        y_hat = utils.to_tensor(y_hat, dtype=dtype, device=device)

        self._dtype = dtype
        self._device = y.device
        if rng is None:
            rng = torch.Generator(device=device)
            rng.seed()
        self._rng = rng

        # check dtypes and devices
        # assert self._jac.dtype == self._dtype
        # assert self._jac.device == self._device

        # compute derivatives of loss function
        (
            self._y,
            self._y_hat,
            self._dloss_dy_hat,
            self._d2loss_dboth,
            self._d2loss_dy_hat2,
        ) = utils.compute_derivatives(self._loss, y, y_hat)

        # precompute some quantities for working with generic form
        # a = dloss_dy_hat
        # b = d2loss_dy_hat2
        # c = d2loss_dboth
        self._a_c = self._dloss_dy_hat / self._d2loss_dy_hat2
        self._c_b = self._d2loss_dy_hat2 / self._d2loss_dboth

        # the normalized Jacobian has diagonal values in the range [0, 1]
        self._normalized_jac = self._jac @ lo.DiagonalOperator(-self._c_b)

        self._n_matvecs = 0
        self._normalized_diag_jac_estims = None
        self._y_tilde_exact = None

    def evaluate(
        self,
        risk_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_matvecs: int = 100,
        subsets: list[list[int]] | int = 50,
    ) -> float:
        """Evaluate the risk function using RandALO:

            R = 1 / n * sum_i risk(y_i, y_hat_i).

        Parameters
        ----------
        risk_fun : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The risk function to evaluate.
        n_matvecs : int, optional
            The number of Jacobian–vector products to compute for the RandALO
            method, by default 100.
        subsets : list[list[int]] | int, optional
            The subsets of Jacobian–vector products to use for the debiasing
            step in RandALO. If an integer is provided, the subsets are chosen
            randomly with sizes uniformly taken between `n_matvecs // 2` and
            `n_matvecs`. By default 50.

        Returns
        -------
        float
            The risk estimate.
        """
        self._do_diag_jac_estims_upto(n_matvecs)

        # compute BKS estimates for subsets of Jacobian–vector products
        if isinstance(subsets, int):
            subsets = [
                torch.randperm(n_matvecs, generator=self._rng)[:m]
                for m in torch.linspace(n_matvecs // 2, n_matvecs, subsets, dtype=int)
            ]
        mixing_matrix = utils.create_mixing_matrix(n_matvecs, subsets)
        mus = self._normalized_diag_jac_estims[:, :n_matvecs] @ mixing_matrix
        m_primes = torch.sum(mixing_matrix > 0, dim=0, keepdim=True)
        normalized_diag_jacs = self._uniform_map_estimates(mus, m_primes)
        y_tildes = [
            self._y_tilde_from_normalized_jac(normalized_diag_jacs[:, j])
            for j in range(len(subsets))
        ]
        risks = [risk_fun(self._y, y_tilde).item() for y_tilde in y_tildes]
        # return utils.robust_y_intercept(1 / m_primes, risks), m_primes, risks
        return utils.robust_y_intercept(1 / m_primes, risks)

    def evaluate_bks(
        self,
        risk_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_matvecs: int = 100,
    ) -> float:
        """Evaluate the risk function using the plug-in BKS estimate.

        Parameters
        ----------
        risk_fun : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The risk function to evaluate.
        n_matvecs : int, optional
            The number of Jacobian–vector products to compute for the BKS
            method, by default 100.

        Returns
        -------
        float
            The risk estimate.
        """

        self._do_diag_jac_estims_upto(n_matvecs)

        mus = self._normalized_diag_jac_estims[:, :n_matvecs].mean(dim=1)
        normalized_diag_jac_bks = self._uniform_map_estimates(
            mus, utils.unsqueeze_scalar_like(n_matvecs, mus)
        )

        y_tilde = self._y_tilde_from_normalized_jac(normalized_diag_jac_bks)
        return risk_fun(self._y, y_tilde).item()

    def evaluate_alo(
        self, risk_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> float:
        """Perform exact ALO.

        Parameters
        ----------
        risk_fun : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The risk function to evaluate.

        Returns
        -------
        float
            The risk estimate.

        """
        if self._y_tilde_exact is None:
            if hasattr(self._jac, "diag"):
                self._y_tilde_exact = self._y_tilde_from_jac(self._jac.diag())
            else:
                self._y_tilde_exact = self._y_tilde_from_normalized_jac(
                    torch.diag(self._normalized_jac @ torch.eye(self._y.shape[0]))
                )

        return risk_fun(self._y, self._y_tilde_exact).item()

    def _y_tilde_from_normalized_jac(
        self, normalized_jac: torch.Tensor
    ) -> torch.Tensor:
        """Compute the corrected predictions using the normalized Jacobian.

        Parameters
        ----------
        normalized_jac : torch.Tensor
            The transformed diagonal of the Jacobian.

        Returns
        -------
        torch.Tensor
            The corrected predictions.
        """
        normalized_jac = torch.clamp(normalized_jac, 0, 1)
        return self._y_hat + self._a_c * normalized_jac / (1 - normalized_jac)

    def _y_tilde_from_jac(self, diag_jac: torch.Tensor) -> torch.Tensor:
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

        return self._y_tilde_from_normalized_jac(-diag_jac * self._c_b)

    def _get_matvecs(self, n_matvecs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute random Jacobian–vector products using random Rademacher
        vectors.

        Parameters
        ----------
        n_matvecs : int
            The number of samples to compute.

        Returns
        -------
        [Tensor, Tensor]
            The matrix–vector products and the random vectors.
        """
        Omega = (
            torch.randint(
                0,
                2,
                (self._y.shape[0], n_matvecs),
                generator=self._rng,
                dtype=self._dtype,
            )
            * 2.0
            - 1
        )
        return self._jac @ Omega, Omega

    def _do_diag_jac_estims_upto(self, n_matvecs: int) -> None:
        """Compute more diagonal Jacobian estimates.

        Parameters
        ----------
        n_matvecs : int
            The desired total number of samples.
        """

        if n_matvecs <= self._n_matvecs:
            return

        # compute the matrix–vector products
        matvecs, Omega = self._get_matvecs(n_matvecs - self._n_matvecs)

        # update the diagonal Jacobian estimates
        normalized_diag_jac_estims = matvecs * Omega
        if self._normalized_diag_jac_estims is None:
            self._normalized_diag_jac_estims = normalized_diag_jac_estims
        else:
            self._normalized_diag_jac_estims = torch.cat(
                (self._normalized_diag_jac_estims, normalized_diag_jac_estims), dim=1
            )
        self._n_matvecs = n_matvecs

        # compute statistics for truncated normal MAP estimation
        self._normalized_diag_jac_stds = self._normalized_diag_jac_estims.std(dim=1)

    def _uniform_map_estimates(self, mus: torch.Tensor, ms: torch.Tensor):
        """Compute MAP estimates with a Uniform[0, 1] prior given sample means.
        Uses the standard deviations computed from all available samples, even
        for sample means computed from a subset of samples.

        Parameters
        ----------
        mus : torch.Tensor
            The sample means of the diagonal estimates.
        ms: torch.Tensor
            The number of samples, broadcastable with `mus`.

        Returns
        -------
        torch.Tensor
            The MAP estimates.
        """
        return truncnorm.truncnorm_mean(
            mus,
            self._normalized_diag_jac_stds.reshape(-1, *([1] * (mus.ndim - 1)))
            / torch.sqrt(ms),
            utils.unsqueeze_scalar_like(0.0, mus),
            utils.unsqueeze_scalar_like(1.0, mus),
        )

    @classmethod
    def from_sklearn(cls, model, X, y):
        sklearn.utils.validation.check_is_fitted(model)

        match model:
            case sklearn.linear_model.LinearRegression():
                loss = ml.MSELoss()
                y = y
                y_hat = model.predict(X)
            case _:
                raise ValueError(f"Model {model.__class__.__name__} not supported.")

        jac = ml.Jacobian()

        return cls(loss=loss, jac=jac, y=y, y_hat=y_hat)
