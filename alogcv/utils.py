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


def logistic_l1(X, y, C):
    n, p = X.shape
    assert y.shape == (n,)
    lambda_val = 1 / C  # Same normalization as sklearn

    def prox(v, t):
        return torch.relu(v - lambda_val * t) - torch.relu(-v - lambda_val * t)

    A = lo.aslinearoperator(X)  # Construct linear operator from 2D tensor.
    y = y.clone()
    y.requires_grad_(True)

    t0 = time.monotonic()
    solver = alogcv.solver.FISTALogistic(
        A, prox, torch.zeros(p).to(X.device), device=y.device
    )
    beta = solver.solve(y)
    tf = time.monotonic()
    #y_hat = torch.sigmoid(A @ beta)
    y_hat = A @ beta
    H = lo.VectorJacobianOperator(y_hat, y)

    return beta, H, tf - t0, solver._iters


def jvp_generalized_hessian(
        X, l_diag, D, r_diag, Z
):
    
    l_finite_mask = torch.isfinite(l_diag)
    r_finite_mask = torch.isfinite(r_diag)
    H = X.T[:, l_finite_mask] @ lo.DiagonalOperator(l_diag[l_finite_mask]) @ X[l_finite_mask, :] + \
        D.T[:, r_finite_mask] @ lo.DiagonalOperator(r_diag[l_finite_mask]) @ D[l_finite_mask, :]
    A = D[~r_finite_mask, :]
    assert torch.sum(l_finite_mask).item() == l_diag.numel()

    top_row = H.cat(A.T, dim=1)
    bottom_row = A.cat(torch.zeros(n := top_row.shape[1] - A.shape[1], n, device=A.device), dim=1)
    M = top_row.cat(bottom_row, dim=0)

    LD, pivots = torch.linalg.ldl_factor(M)

    return X @ linalg.ldl_solve(LD, pivots, X.T @ Z)
