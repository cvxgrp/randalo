import numpy as np
import torch
from sklearn.linear_model import Lasso
from torch import autograd, Tensor
from tqdm import tqdm

from alogcv.alo import ALOExact, ALORandomized

shapes = [(2000, 1800), (1000, 2000)]
#shapes = [(2000, 1000), (2000, 1800), (1000, 2000)]
lasso_lamdas = np.logspace(-2.5, 0, 30)
ridge_lamdas = np.logspace(-3, 1, 30)
n_trials = 10
ms = range(30, 200, 30)
sigma = 1
device = "cpu"

torch.manual_seed(0xEE_364_AB)


def loss_fun(y, y_hat):
    return (y - y_hat) ** 2 / 2


def risk(y, y_hat):
    return (y - y_hat) ** 2


data = {}
exacts = {}
for n, p in shapes:
    # Lasso
    beta = torch.randn(p, device=device) / np.sqrt(p // 3)
    beta[p // 3 :] = 0
    nu = 5
    X = torch.distributions.studentT.StudentT(nu).sample((n, p))
    # X =  (torch.distributions.exponential.Exponential(1.0).sample((n,)) + 0.05)[:, None] * \
    #       torch.distributions.normal.Normal(0, 1).sample((n, p))
    # X = torch.distributions.normal.Normal(0, 1).sample((n, p))
    mu = X @ beta
    # y = mu + torch.distributions.laplace.Laplace(0, 1).rsample((n,))
    y = mu + torch.distributions.normal.Normal(0, 1).rsample((n,))

    for lamda in lasso_lamdas:
        lasso = Lasso(lamda)
        lasso.fit(X, y)
        beta_hat = Tensor(lasso.coef_)

        mask = torch.abs(beta_hat) > 1e-8

        X_lasso = X[:, mask]
        Q, _ = torch.linalg.qr(X_lasso)
        H = Q @ Q.T
        h = torch.diag(H)
        y_hat = X @ beta_hat
        exacts[((n, p), "lasso", lamda)] = (exact := ALOExact(loss_fun, y, y_hat, h))
        np.savez(f'data/n{n}-p{p}-lasso-lamda{lamda}-exacts.npz',
            exact=np.array(exact.eval_risk(risk)),
        )


        for m in ms:
            randoms = [ALORandomized(loss_fun, y, y_hat, H, m) for _ in range(n_trials)]
            values = np.array([r.eval_risk(risk) for r in randoms])
            np.savez(f'data/n{n}-p{p}-lasso-lamda{lamda}-m{m}.npz',
                     values=values,
                     ms=np.vstack([r._ms for r in randoms]),
                     risks=np.vstack([r._risks for r in randoms]),
                     res_m_to_risk_fit=np.vstack([r._res_m_to_risk_fit for r in randoms]),
            )
        data[((n, p), "lasso", lamda, m)] = randoms


    X = torch.randn(n, p, device=device)
    beta = torch.randn(p, device=device) / np.sqrt(p)
    y = X @ beta + torch.randn(n, device=device) * sigma

    for lamda in ridge_lamdas:
        beta_hat = torch.linalg.solve(
            X.T @ X + n * lamda * torch.eye(p, device=device), X.T @ y
        )
        H = X @ torch.linalg.solve(
            X.T @ X + n * lamda * torch.eye(p, device=device), X.T
        )
        h = torch.diag(H)
        y_hat = X @ beta_hat

        exacts[((n, p), "ridge", lamda)] = (exact := ALOExact(loss_fun, y, y_hat, h))
        np.savez(f'data/n{n}-p{p}-ridge-lamda{lamda}-exacts.npz',
            exact=np.array(exact.eval_risk(risk)),
        )

        for m in ms:
            randoms = [ALORandomized(loss_fun, y, y_hat, H, m) for _ in range(n_trials)]
            values = np.array([r.eval_risk(risk) for r in randoms])
            np.savez(f'data/n{n}-p{p}-ridge-lamda{lamda}-m{m}.npz',
                     values=values,
                     ms=np.vstack([r._ms for r in randoms]),
                     risks=np.vstack([r._risks for r in randoms]),
                     res_m_to_risk_fit=np.vstack([r._res_m_to_risk_fit for r in randoms]),
            )

        data[((n, p), "ridge", lamda, m)] = randoms

