import cvxpy as cp
import numpy as np
import torch
from torch import Tensor


def robust_poly_fit(x, y, order: int):
    beta = cp.Variable(order + 1)
    r = y - np.vander(x, order + 1, True) @ beta

    prob = cp.Problem(cp.Minimize(cp.sum(cp.huber(r))))
    if not (prob.solve(cp.CLARABEL) < np.inf):
        return [np.nan for _ in range(order + 1)], np.inf
    return beta.value, np.linalg.norm(r.value)


def torch_normal_pdf(x: Tensor) -> Tensor:
    """Compute the probability density function of the standard normal distribution."""
    return torch.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)


def torch_normal_cdf(x: Tensor) -> Tensor:
    """Compute the cumulative distribution function of the standard normal distribution."""
    return 0.5 * (1 + torch.erf(x / np.sqrt(2)))


def truncated_normal_mean(
    mu: Tensor, sigma: Tensor, a: float = 0, b: float = 1
) -> Tensor:
    """Compute the means of truncated normal distributions.

    Parameters
    ----------
    mu : Tensor
        Means.
    sigma : Tensor
        Standard deviations.
    a : float
        Lower truncation point.
    b : float
        Upper truncation point.

    Returns
    -------
    Tensor
        Means of truncated normal distributions.
    """

    # see https://en.wikipedia.org/wiki/Truncated_normal_distribution#Moments
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    Z = torch_normal_cdf(beta) - torch_normal_cdf(alpha)
    mu_tilde = mu + sigma * (torch_normal_pdf(alpha) - torch_normal_pdf(beta)) / Z

    # in case of division by zero, return clamped mean
    mu_tilde[torch.isnan(mu_tilde)] = torch.clamp(mu[torch.isnan(mu_tilde)], a, b)
    return mu_tilde
