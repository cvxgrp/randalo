from typing import Callable, Optional

import numpy as np
from numpy.polynomial.polynomial import Polynomial

import torch
from torch import autograd, Tensor

import linops as lo
from linops import LinearOperator

import utils

# Loss function: is a function from R^n \times R^n to R^n that operates on pairs

class ALOBase(object):
    def __init__(
        self, loss_fun: Callable[[Tensor, Tensor], Tensor], y: Tensor, y_hat: Tensor
    ):
        self._loss_fun = loss_fun

        self._y = y.detach().clone().requires_grad_(True)
        self._y_hat = y_hat.detach().clone().requires_grad_(True)
        self.n = self._y.shape[0]

        loss = loss_fun(self._y, self._y_hat).sum()
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

    def y_tilde(self, diag_jac: Tensor, epsilon=1e-8) -> Tensor:
        diag_jac = (
            torch.clamp(diag_jac, epsilon, 1 - epsilon)
            if epsilon is not None
            else diag_jac
        )
        return self._y_hat - self._dloss_dy_hat * diag_jac / (
            self._d2loss_dboth + self._d2loss_dy_hat2 * diag_jac
        )


class ALOExact(ALOBase):
    def __init__(
        self,
        loss_fun: Callable[[Tensor, Tensor], Tensor],
        y: Tensor,
        y_hat: Tensor,
        diag_jac: Tensor,
    ):
        super().__init__(loss_fun, y, y_hat)
        self._diag_jac = diag_jac
        self._y_tilde = self.y_tilde(self._diag_jac)

    def joint_vars(self) -> [Tensor, Tensor]:
        return self._y, self._y_tilde

    def eval_risk(self, risk: Callable[[Tensor, Tensor], Tensor]) -> float:
        return risk(self._y, self._y_tilde).sum().item()


class ALOBKS(ALOBase):
    def __init__(
        self,
        loss_fun: Callable[[Tensor, Tensor], Tensor],
        y: Tensor,
        y_hat: Tensor,
        jac: LinearOperator,
        m: int,
        generator: torch.Generator = None,
    ):
        super().__init__(loss_fun, y, y_hat)
        self._jac = jac

        self._diag_jac_estims = None
        self.m = 0
        self._generator = generator

        self._best_diag_jac = None
        self.do_more_diag_jac_estims(m)

    def _get_matvecs(self, m: int) -> [Tensor, Tensor]:
        Omega = torch.randint(0, 2, (self.n, m), generator=self._generator) * 2.0 - 1
        return self._jac @ Omega, Omega

    def do_more_diag_jac_estims(self, m: int) -> None:
        matvecs, Omega = self._get_matvecs(m)

        diag_jac_estims = matvecs * Omega
        if self._diag_jac_estims is None:
            self._diag_jac_estims = diag_jac_estims
        else:
            self._diag_jac_estims = torch.cat(
                (self._diag_jac_estims, diag_jac_estims), dim=1
            )
        self.m += m

        self._best_diag_jac = self._diag_jac_estims.mean(dim=1)

    def joint_vars(self) -> [Tensor, Tensor]:
        return self._y, self.y_tilde(self._best_diag_jac)

    def eval_risk(
        self,
        risk: Callable[[Tensor, Tensor], Tensor],
        order: Optional[int] = 1,
        power: float = 1.0,
    ) -> float:
        if order is None:
            return risk(self._y, self.y_tilde(self._best_diag_jac)).sum().item()
        else:
            assert self.m > 1
            m0 = self.m // 2

            xs = np.zeros(50)
            ys = np.zeros(50)

            it = iter(enumerate(np.linspace(1, self.m - m0 - 1, 50).astype(int)))
            total_attempts = 0
            finished = True
            while True:
                total_attempts += 1
                if total_attempts > 75:
                    print(f'Failed! {self.m=}')
                    return np.nan
                if finished:
                    try:
                        idx, i = next(it)
                    except StopIteration:
                        break
                    finished = False
                m = m0 + i
                xs[idx] = 1 / m**power
                # diag_jac = (diag_jac * (m - 1) + self._diag_jac_estims[:, m - 1]) / m
                diag_jac = self._diag_jac_estims[
                    :, np.random.choice(self.m, m, replace=False)
                ].mean(dim=1)
                if (self._d2loss_dboth + self._d2loss_dy_hat2 * diag_jac >= 0).any():
                    continue

                ys[idx] = risk(self._y, self.y_tilde(diag_jac)).sum().item()
                finished = True

            coefs, residual_norm = utils.robust_poly_fit(xs, ys, order)

            return coefs[0]
