import torch
from torch import autograd, Tensor

torch.set_default_dtype(torch.float64)

from sklearn.linear_model import LogisticRegression

import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt


from alogcv.alo import ALOExact, ALORandomized
import alogcv.utils

n = 2000
p = 1800
sigma = 1
lamdas = np.logspace(-1.9, -0.9, 30)
ms = [30, 50]
n_trials = 2
device = "cpu"

beta = torch.ones(p, device=device) / np.sqrt(p // 3)
beta[p // 3 :] = 0


def generate_sample():
    nu = 5
    _X = torch.distributions.studentT.StudentT(nu).sample((n, p))
    # _X = (torch.distributions.exponential.Exponential(1.0).sample((n,)) + 0.05)[
    #    :, None
    # ] * torch.distributions.normal.Normal(0, 1).sample((n, p))
    # _X = torch.distributions.normal.Normal(0, 1).sample((n, p))
    _logits = _X @ beta
    # y = _mu + torch.distributions.laplace.Laplace(0, 1).rsample((n,))
    _y = torch.distributions.bernoulli.Bernoulli(logits=_logits).sample()
    return _X, _y


X, y = generate_sample()
X_test, y_test = generate_sample()


def loss_fun(y, y_hat):
    return (y - y_hat) ** 2 / 2


def risk(y, y_hat):
    return (y - y_hat) ** 2


risks_oos = np.zeros(len(lamdas))
risks_alo = np.zeros(len(lamdas))
risks_loo_shortcut = np.zeros(len(lamdas))
risks_bks = np.zeros((len(lamdas), len(ms), n_trials))
risks_poly = np.zeros((len(lamdas), len(ms), n_trials))

# var_bks = np.zeros((len(lamdas), len(ms), n_trials))
# oneminushoverstdbks = np.zeros((len(lamdas), len(ms), n_trials))

for i, lamda in enumerate(tqdm(lamdas)):
    C = 1 / lamda / n
    logreg = LogisticRegression("l1", C=C, fit_intercept=False, solver="saga")
    logreg.fit(X, y)
    beta_hat = Tensor(logreg.coef_.squeeze())
    y_hat = torch.sigmoid(X @ beta_hat)
    y_hat_test = torch.sigmoid(X_test @ beta_hat)
    risks_oos[i] = risk(y_test, y_hat_test).sum() / n

    beta_est, Hest, tdur, iters = alogcv.utils.logistic_l1(X, y, C)
    print(f"{iters=} in {tdur=}s")
    print(f"{torch.linalg.norm(beta_hat - beta_est)/torch.linalg.norm(beta_hat)=}")

    for j, m in enumerate(ms):
        for trial in range(n_trials):
            # alo_bks = ALOBKS(loss_fun, y, y_hat, H, m)
            alo_bks = ALORandomized(loss_fun, y, y_hat, Hest, m)
            # alo_bks = ALORandomized(loss_fun, y, y_hat, H, m)

            # alo_bks = ALOBKSWithMultiplicativeErrorBounds(loss_fun, y, y_hat, H, m)
            risks_bks[i, j, trial] = alo_bks.eval_risk(risk, order=None) / n
            risks_poly[i, j, trial] = alo_bks.eval_risk(risk, order=1) / n
    # print(alo_bks.eval_risk(risk, order=None))
    # print(alo_bks.eval_risk(risk, order=1))

plt.plot(lamdas, risks_oos, ":k", label="OOS")
# plt.plot(lamdas, risks_alo, "k", label="alo")
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

plt.title(f"ALO for Logistic LASSO Regression, ${n=}$, ${p=}$, $\\sigma={sigma}$")
plt.xlabel("$\lambda$")
plt.ylabel("Risk")

plt.show()
