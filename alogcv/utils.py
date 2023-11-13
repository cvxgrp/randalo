import time

import cvxpy as cp
import numpy as np
import linops as lo
import torch

import alogcv.solver


def robust_poly_fit(x, y, order: int):
    beta = cp.Variable(order + 1)
    r = y - np.vander(x, order + 1, True) @ beta

    prob = cp.Problem(cp.Minimize(cp.sum(cp.huber(r))))
    if not (prob.solve(cp.CLARABEL) < np.inf):
        return [np.nan for _ in range(order + 1)], np.inf
    return beta.value, r.value


def lasso(X, y, lamda):
    n, p = X.shape
    assert y.shape == (n,)
    lambda_val = lamda * n  # Same normalization as sklearn

    def prox(v, t):
        return torch.relu(v - lambda_val * t) - torch.relu(-v - lambda_val * t)

    A = lo.aslinearoperator(X)  # Construct linear operator from 2D tensor.
    y = y.clone()
    y.requires_grad_(True)

    t0 = time.monotonic()
    solver = alogcv.solver.FISTASolver(
        A, prox, torch.zeros(p).to(X.device), device=y.device
    )
    beta = solver.solve(y)
    tf = time.monotonic()
    y_hat = A @ beta
    H = lo.VectorJacobianOperator(y_hat, y)

    return beta, H, tf - t0, solver._iters
