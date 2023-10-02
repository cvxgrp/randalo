from typing import Callable, Optional

import numpy as np
from numpy.polynomial.polynomial import Polynomial

import torch
from torch import autograd, Tensor

import linops as lo
from linops import LinearOperator


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

            xs = 1 / np.arange(m0, self.m) ** power
            ys = np.zeros_like(xs)
            diag_jac = self._diag_jac_estims[:, :m0].mean(dim=1)
            ys[0] = risk(self._y, self.y_tilde(diag_jac)).sum().item()

            for i in np.linspace(1, self.m - m0 - 1, 50).astype(int):
                m = m0 + i
                # diag_jac = (diag_jac * (m - 1) + self._diag_jac_estims[:, m - 1]) / m
                diag_jac = self._diag_jac_estims[
                    :, np.random.choice(self.m, m, replace=False)
                ].mean(dim=1)
                ys[i] = risk(self._y, self.y_tilde(diag_jac)).sum().item()

            domain = [xs[0], xs[-1]]
            poly, (resid, *_) = Polynomial.fit(
                xs,
                ys,
                deg=order,
                w=1 / xs**0,
                full=True,
                domain=domain,
                window=domain,
            )
            yys = poly(xs)
            # print(1 - ((yys - ys) ** 2 @ (np.ones(len(ys)))) / (np.var(ys) * len(ys)))
            return poly.coef[0]


if __name__ == "__main__":
    from tqdm import tqdm
    from matplotlib import pyplot as plt

    n = 2000
    p = 1000
    sigma = 1
    lamdas = np.logspace(-3, 1, 30)
    ms = [30, 50, 100]
    n_trials = 2
    device = "cpu"

    X = torch.randn(n, p, device=device)
    beta = torch.randn(p, device=device) / np.sqrt(p)
    y = X @ beta + torch.randn(n, device=device) * sigma

    def loss_fun(y, y_hat):
        return (y - y_hat) ** 2 / 2

    def risk(y, y_hat):
        return (y - y_hat) ** 2

    risks_gen = np.zeros(len(lamdas))
    risks_alo = np.zeros(len(lamdas))
    risks_loo_shortcut = np.zeros(len(lamdas))
    risks_bks = np.zeros((len(lamdas), len(ms), n_trials))
    risks_poly = np.zeros((len(lamdas), len(ms), n_trials))

    for i, lamda in enumerate(tqdm(lamdas)):
        beta_hat = torch.linalg.solve(
            X.T @ X + n * lamda * torch.eye(p, device=device), X.T @ y
        )
        H = X @ torch.linalg.solve(
            X.T @ X + n * lamda * torch.eye(p, device=device), X.T
        )
        h = torch.diag(H)

        y_hat = X @ beta_hat

        risks_gen[i] = risk(beta, beta_hat).sum().item() + sigma**2

        alo_exact = ALOExact(loss_fun, y, y_hat, h)
        risks_alo[i] = alo_exact.eval_risk(risk) / n
        risks_loo_shortcut[i] = torch.sum((y - y_hat) ** 2 / (1 - h) ** 2).item() / n

        for j, m in enumerate(ms):
            for trial in range(n_trials):
                alo_bks = ALOBKS(loss_fun, y, y_hat, H, m)
                risks_bks[i, j, trial] = alo_bks.eval_risk(risk, order=None) / n
                risks_poly[i, j, trial] = alo_bks.eval_risk(risk, order=1) / n
        # print(alo_bks.eval_risk(risk, order=None))
        # print(alo_bks.eval_risk(risk, order=1))

    plt.plot(lamdas, risks_gen, ":k", label="gen")
    plt.plot(lamdas, risks_alo, "k", label="alo")
    # plt.plot(lamdas, risks_loo_shortcut, label="loo_shortcut")

    ylim = plt.ylim()

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j, m in enumerate(ms):
        # errorbar
        plt.errorbar(
            lamdas,
            risks_bks[:, j, :].mean(axis=1),
            yerr=risks_bks[:, j, :].std(axis=1),
            label=f"bks_{m}",
            color=color_cycle[j],
        )
        plt.errorbar(
            lamdas,
            risks_poly[:, j, :].mean(axis=1),
            yerr=risks_poly[:, j, :].std(axis=1),
            label=f"poly_{m}",
            linestyle="dashed",
            color=color_cycle[j],
        )

    plt.ylim(ylim[0] * 0.9, ylim[1] * 1.1)

    plt.legend()
    plt.xscale("log")

    plt.title(f"ALO for Ridge Regression, ${n=}$, ${p=}$, $\\sigma={sigma}$")
    plt.xlabel("$\lambda$")
    plt.ylabel("Risk")

    plt.show()
