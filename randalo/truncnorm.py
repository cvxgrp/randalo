import math

import torch

"""Implementation of truncated normal distributions in PyTorch.

This module is based on the implementation in scipy.stats.truncnorm, which is
licensed under the BSD license. The original source code can be found at
https://github.com/scipy/scipy/blob/v1.11.3/scipy/stats/_continuous_distns.py

Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers.
All rights reserved.
"""

# Constants for the standard normal distribution
_norm_pdf_C = math.sqrt(2 * math.pi)
_norm_pdf_logC = math.log(_norm_pdf_C)


def _norm_logpdf(x: torch.Tensor) -> torch.Tensor:
    """Log of the probability density function of the standard normal distribution."""
    return -(x**2) / 2.0 - _norm_pdf_logC


def _log_diff(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """Compute log(a - b) when log_a > log_b and both are log-space values"""
    return log_a + torch.log1p(-torch.exp(log_b - log_a))


def _log_gauss_mass(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Log of Gaussian probability mass within an interval"""

    alpha, beta = torch.broadcast_tensors(alpha, beta)

    # Calculations in right tail are inaccurate, so we'll exploit the
    # symmetry and work only in the left tail
    case_left = beta <= 0
    case_right = alpha > 0
    case_central = ~(case_left | case_right)

    def mass_case_left(a, b):
        return _log_diff(torch.special.log_ndtr(b), torch.special.log_ndtr(a))

    def mass_case_right(a, b):
        return mass_case_left(-b, -a)

    def mass_case_central(a, b):
        return torch.log1p(-torch.special.ndtr(a) - torch.special.ndtr(-b))

    # Initialize tensor for output
    out = torch.full_like(alpha, float("nan"))
    out[case_left] = mass_case_left(alpha[case_left], beta[case_left]).real
    out[case_right] = mass_case_right(alpha[case_right], beta[case_right]).real
    out[case_central] = mass_case_central(alpha[case_central], beta[case_central])

    return out


def _truncnorm_log_pdf(x, alpha, beta):
    """Log of the probability density function of the truncated normal distribution."""
    return _norm_logpdf(x) - _log_gauss_mass(alpha, beta)


def _truncnorm_pdf(x, alpha, beta):
    """Probability density function of the truncated normal distribution."""
    return torch.exp(_truncnorm_log_pdf(x, alpha, beta))


def truncnorm_mean(
    mu: torch.Tensor, sigma: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Calculates the mean of a truncated normal distribution given its parameters.

    Parameters
    ----------
    mu : torch.Tensor
        The means of the normal distributions
    sigma : torch.Tensor
        The standard deviations of the normal distributions
    a : torch.Tensor
        The lower bounds of the truncation interval
    b : torch.Tensor
        The upper bounds of the truncation interval

    Returns
    -------
    torch.Tensor
        The means of the truncated normal distributions
    """

    mu, sigma, a, b = torch.broadcast_tensors(mu, sigma, a, b)

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    nan = torch.isnan(alpha) | torch.isnan(beta)
    notnan = ~nan
    alpha = alpha[notnan]
    beta = beta[notnan]
    alpha_beta = torch.stack([alpha, beta], dim=0)
    pA, pB = _truncnorm_pdf(alpha_beta, alpha[None, :], beta[None, :])

    out = torch.full_like(mu, float("nan"))
    out[nan] = torch.clamp(mu[nan], min=a[nan], max=b[nan])
    out[notnan] = mu[notnan] - sigma[notnan] * (pB - pA)

    return out
