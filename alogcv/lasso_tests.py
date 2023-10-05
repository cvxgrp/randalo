import torch
from torch import autograd, Tensor
torch.set_default_dtype(torch.float64)

from sklearn.linear_model import Lasso

import numpy as np 

from tqdm import tqdm
from matplotlib import pyplot as plt


from alo import ALOExact, ALOBKS

n = 1000
p = 2000
sigma = 1
lamdas = np.logspace(-2.5, 0, 30)
ms = [30, 50, 100]
n_trials = 2
device = "cpu"

beta = torch.randn(p, device=device) / np.sqrt(p // 3)
beta[p // 3:] = 0

def generate_sample():
    nu = 5
    #_X = torch.distributions.studentT.StudentT(nu).sample((n, p))
    _X =  (torch.distributions.exponential.Exponential(1.0).sample((n,)) + 0.05)[:, None] * \
           torch.distributions.normal.Normal(0, 1).sample((n, p))
    #_X = torch.distributions.normal.Normal(0, 1).sample((n, p))
    _mu = _X @ beta
    #y = _mu + torch.distributions.laplace.Laplace(0, 1).rsample((n,))
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
risks_bks = np.zeros((len(lamdas), len(ms), n_trials))
risks_poly = np.zeros((len(lamdas), len(ms), n_trials))

for i, lamda in enumerate(tqdm(lamdas)):
    lasso = Lasso(lamda)
    lasso.fit(X, y)
    beta_hat = Tensor(lasso.coef_)

    mask = torch.abs(beta_hat) > 1e-8
    print(mask.sum().item())

    X_lasso = X[:, mask]
    Q, _ = torch.linalg.qr(X_lasso)
    H = Q @ Q.T
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

plt.title(f"ALO for LASSO Regression, ${n=}$, ${p=}$, $\\sigma={sigma}$")
plt.xlabel("$\lambda$")
plt.ylabel("Risk")

plt.show()

