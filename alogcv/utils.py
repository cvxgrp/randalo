import time

import cvxpy as cp
import numpy as np
import linops as lo
import scipy
import torch
from torch import autograd

import alogcv.solver


def robust_poly_fit(x, y, order: int):
    beta = cp.Variable(order + 1)
    r = y - np.vander(x, order + 1, True) @ beta

    prob = cp.Problem(cp.Minimize(cp.sum(cp.huber(r))))
    if not (prob.solve(cp.CLARABEL) < np.inf):
        return [np.nan for _ in range(order + 1)], np.inf
    return beta.value, r.value

def weighted_lstsq_fit(x, y, order: int, cov):
    # solves min_beta  1/2 (X beta - y)^T cov^{-1} (X @ beta - y)
    #                = 1/2 (beta^T X^T - y^T) (cov^{-1} X @ beta - cov^{-1} y)
    #                = 1/2 (beta^T X^T cov^{-1} X @ beta) - beta^T X^T cov^{-1} y + constant
    #                = 1/2 (beta^T X^T cov^{-1} X @ beta) - beta^T X^T cov^{-1} y + constant
    #   i.e. beta^star = Ly = (X^T cov^{-1} X)^{-1} X^T cov^{-1} y
    # Alternatively we're after 
    #  min. 1/2 r^T cov^{-1} r s.t. r = y - X beta
    # which becomes
    # min. 1/2 r^T cov^{-1} r s.t. y = [I X] (r, beta)
    # [cov^{-1} 0 I  ] [r   ]   [0]
    # [0        0 X^T] [beta] = [0]
    # [I        X 0  ] [nu  ]   [y]
    # whose Schur complement is given by
    # ([0 X^T]   [ 0 ]            )[beta]   [0]
    # ([X 0  ] - [ I ] cov [ 0 I ])[nu  ] = [y]
    # r = -cov [ 0  I ] (beta, nu)
    #
    # [0  X^T][beta]   [0]
    # [X -cov][nu  ] = [y]
    # cannot apply a similar trick here :( have to factor this directly

    # Disabled weighted least squares because of numerical issues for now
    X = np.vander(x, order + 1, True)
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    #L = np.linalg.solve(X.T @ X, X.T)
    #w2 = L @ y

    #Sigma = L @ cov @ L.T
    #stddev = np.sqrt(Sigma[0, 0])

    return w[0]#, stddev

def compute_derivatives(loss_fun, y, y_hat):

    # detach and clone to avoid memory leaks
    y = y.detach().clone().requires_grad_(True)
    y_hat = y_hat.detach().clone().requires_grad_(True)
    n = y.shape[0]

    # compute first and second derivatives of loss function
    # we obtain the vector derivatives by summing and then taking the gradient
    loss = loss_fun(y, y_hat).sum()
    # keep the graph for computing the second derivatives
    dloss_dy_hat, *_ = autograd.grad(loss, y_hat, create_graph=True)
    dloss_dy_hat_sum = dloss_dy_hat.sum()

    d2loss_dboth, d2loss_dy_hat2 = autograd.grad(
        dloss_dy_hat_sum, [y, y_hat], allow_unused=True
    )
    if d2loss_dboth is None:
        d2loss_dboth = torch.zeros_like(y)
    if d2loss_dy_hat2 is None:
        d2loss_dy_hat2 = torch.zeros_like(y_hat)

    # free GPU memory used by autograd and return
    return (
        y.detach(),
        y_hat.detach(),
        dloss_dy_hat.detach(),
        d2loss_dboth.detach(),
        d2loss_dy_hat2.detach(),
    )


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
    # y_hat = torch.sigmoid(A @ beta)
    y_hat = A @ beta
    H = lo.VectorJacobianOperator(y_hat, y)

    return beta, H, tf - t0, solver._iters


class GeneralizedHessianOperator(lo.LinearOperator):
    supports_operator_matrix = True

    def __init__(self, X, l_diag, D, r_diag):
        self._n, self._p = X.shape
        self._shape = (self._n, self._n)
        self._adjoint = self  # Check this assumption
        self._X = X

        l_finite_mask = torch.isfinite(l_diag)
        r_finite_mask = torch.isfinite(r_diag)
        H = (
            X.T[:, l_finite_mask]
            @ lo.DiagonalOperator(l_diag[l_finite_mask])
            @ X[l_finite_mask, :]
            + D.T[:, r_finite_mask]
            @ lo.DiagonalOperator(r_diag[r_finite_mask])
            @ D[r_finite_mask, :]
        )
        A = D[~r_finite_mask, :]
        assert torch.sum(l_finite_mask).item() == l_diag.numel()

        top_row = torch.cat([H, A.T], dim=1)
        self._zero_pad = top_row.shape[1] - A.shape[1]
        bottom_row = torch.cat(
            [A, torch.zeros(self._zero_pad, self._zero_pad, device=A.device)], dim=1
        )
        M = torch.cat([top_row, bottom_row], dim=0)

        self._LD, self._pivots = torch.linalg.ldl_factor(M)

    def _matmul_impl(self, v):
        RHS = torch.cat(
            [
                self._X.T @ v,
                torch.zeros((self._zero_pad, *v.shape[1:]), device=v.device),
            ]
        )
        return (
            self._X @ (torch.linalg.ldl_solve(self._LD, self._pivots, RHS))[: self._p]
        )


def jvp_generalized_hessian(X, l_diag, D, r_diag, Z):
    l_finite_mask = torch.isfinite(l_diag)
    r_finite_mask = torch.isfinite(r_diag)
    H = (
        X.T[:, l_finite_mask]
        @ lo.DiagonalOperator(l_diag[l_finite_mask])
        @ X[l_finite_mask, :]
        + D.T[:, r_finite_mask]
        @ lo.DiagonalOperator(r_diag[l_finite_mask])
        @ D[l_finite_mask, :]
    )
    A = D[~r_finite_mask, :]
    assert torch.sum(l_finite_mask).item() == l_diag.numel()

    top_row = H.cat(A.T, dim=1)
    bottom_row = A.cat(
        torch.zeros(n := top_row.shape[1] - A.shape[1], n, device=A.device), dim=1
    )
    M = top_row.cat(bottom_row, dim=0)

    LD, pivots = torch.linalg.ldl_factor(M)

    return X @ linalg.ldl_solve(LD, pivots, X.T @ Z)
