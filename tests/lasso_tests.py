import torch
from torch import autograd, Tensor

torch.set_default_dtype(torch.float64)

from sklearn.linear_model import Lasso

import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt


from alogcv.alo import ALOExact, ALORandomized
import alogcv.utils

torch.manual_seed(0x364ef)

n = 300
s = 100
p = 5 * s
sigma = 1
lamdas = np.logspace(-2, 0, 30)
ms = [30]
n_trials = 1
device = "cpu"

beta = torch.randn(p, device=device) / np.sqrt(s)
beta[s:] = 0


def generate_sample():
    nu = 5
    # _X = torch.distributions.studentT.StudentT(nu).sample((n, p))
    #_X = (torch.distributions.exponential.Exponential(1.0).sample((n,)) + 0.05)[
    #    :, None
    #] * torch.distributions.normal.Normal(0, 1).sample((n, p))
    _X = torch.distributions.normal.Normal(0, 1).sample((n, p))
    _mu = _X @ beta
    # y = _mu + torch.distributions.laplace.Laplace(0, 1).rsample((n,))
    _y = _mu + torch.distributions.normal.Normal(0, 1).rsample((n,))
    return _X, _mu, _y


X, mu, y = generate_sample()


def loss_fun(y, y_hat):
    return (y - y_hat) ** 2 / 2


def risk(y, y_hat):
    return (y - y_hat) ** 2


risks_gen = np.zeros(len(lamdas))
risks_alo = np.zeros(len(lamdas))
risks_loo_shortcut = np.zeros(len(lamdas))
risks_qp = np.zeros((len(lamdas), len(ms), n_trials))
risks_cf_hessian = np.zeros((len(lamdas), len(ms), n_trials))

for i, lamda in enumerate(tqdm(lamdas)):
    lasso = Lasso(lamda)
    lasso.fit(X, y)
    beta_hat = Tensor(lasso.coef_)

    mask = torch.abs(beta_hat) > 1e-8
    print("non-zero elements: ", mask.sum().item())

    X_lasso = X[:, mask]
    Q, _ = torch.linalg.qr(X_lasso)
    H = Q @ Q.T
    h = torch.diag(H)

    inf_mask = torch.zeros(p)
    inf_mask[~mask] = torch.inf
    H_qp = alogcv.utils.GeneralizedHessianOperator(X, torch.ones(n), torch.eye(p), inf_mask)

    y_hat = X @ beta_hat

    risks_gen[i] = risk(beta, beta_hat).sum().item() + sigma**2

    alo_exact = ALOExact(loss_fun, y, y_hat, h)

    risks_alo[i] = alo_exact.eval_risk(risk) / n
    risks_loo_shortcut[i] = torch.sum((y - y_hat) ** 2 / (1 - h) ** 2).item() / n
    y_hat = X @ beta_hat.detach()

    for j, m in enumerate(ms):
        for trial in range(n_trials):
            gen = torch.Generator()
            gen.manual_seed(1)
            alo_cf_hessian = ALORandomized(loss_fun, y, y_hat, H, m, generator=gen)

            risks_cf_hessian[i, j, trial] = alo_cf_hessian.eval_risk(risk, order=1) / n

            gen = torch.Generator()
            gen.manual_seed(1)

            alo_qp = ALORandomized(loss_fun, y, y_hat, H_qp, m, generator=gen)
            risks_qp[i, j, trial] = alo_qp.eval_risk(risk, order=1) / n

plt.plot(lamdas, risks_gen, ":k", label="gen")
plt.plot(lamdas, risks_alo, "k", label="alo")

ylim = plt.ylim()
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for j, m in enumerate(ms):
    # errorbar
    plt.errorbar(
        lamdas,
        risks_qp[:, j, :].mean(axis=1),
        yerr=risks_qp[:, j, :].std(axis=1),
        label=f"qp_{m}",
        color=color_cycle[j],
    )
    plt.errorbar(
        lamdas,
        risks_cf_hessian[:, j, :].mean(axis=1),
        yerr=risks_cf_hessian[:, j, :].std(axis=1),
        label=f"cf_hessian_{m}",
        linestyle="dashed",
        color=color_cycle[j],
    )

plt.ylim(ylim[0] * 0.9, ylim[1] * 1.1)

plt.legend()
plt.xscale("log")

plt.title(f"ALO for LASSO Regression, ${n=}$, ${p=}$, $\\sigma={sigma}$")
plt.xlabel("$\lambda$")
plt.ylabel("Risk")

plt.show()
