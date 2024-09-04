import unittest

import numpy as np
import scipy
import torch

from randalo import truncnorm


class TruncnormData(object):
    def __init__(self, mu, sigma, a=None, b=None, alpha=None, beta=None):
        self.mu = mu
        self.sigma = sigma

        match (a, b, alpha, beta):
            case (None, None, None, None):
                self.a = -np.inf
                self.b = np.inf
                self.alpha = -np.inf
                self.beta = np.inf

            case (a, b, None, None):
                self.a = a
                self.b = b
                self.alpha = (a - mu) / sigma
                self.beta = (b - mu) / sigma

            case (None, None, alpha, beta):
                self.alpha = alpha
                self.beta = beta
                self.a = mu + alpha * sigma
                self.b = mu + beta * sigma

            case (a, b, alpha, beta):
                raise ValueError("Cannot specify both a, b and alpha, beta")

    def get_torch_tensors(self, dtype=torch.float64):
        return (
            torch.tensor(x, dtype=dtype) for x in [self.mu, self.sigma, self.a, self.b]
        )


class TestTruncnorm(unittest.TestCase):
    def test_truncnorm_mean(self):
        rng = np.random.default_rng(0)

        # test basic truncnorm_mean
        td = TruncnormData(mu=1.0, sigma=1.0, a=-1.0, b=1.0)
        mean_scipy = scipy.stats.truncnorm.mean(
            td.alpha, td.beta, loc=td.mu, scale=td.sigma
        )
        mean_torch = truncnorm.truncnorm_mean(*td.get_torch_tensors()).item()
        self.assertAlmostEqual(mean_scipy, mean_torch, places=10)

        # test truncnorm_mean with no truncation
        td = TruncnormData(mu=1.0, sigma=1.0)
        mean_scipy = scipy.stats.truncnorm.mean(
            td.alpha, td.beta, loc=td.mu, scale=td.sigma
        )
        mean_torch = truncnorm.truncnorm_mean(*td.get_torch_tensors()).item()
        self.assertAlmostEqual(td.mu, mean_torch, places=10)
        self.assertAlmostEqual(mean_scipy, mean_torch, places=10)

        # test truncnorm_mean with broadcasting
        n = 5
        m = 6
        k = 7
        mu = rng.uniform(-1, 1, size=(n, m, k))
        sigma = rng.uniform(0.1, 1, size=(n, m, 1))
        a = rng.uniform(-2, -1, size=(n, 1, k))
        b = rng.uniform(1, 2, size=(1, m, 1))
        td = TruncnormData(mu=mu, sigma=sigma, a=a, b=b)
        mean_scipy = scipy.stats.truncnorm.mean(
            td.alpha, td.beta, loc=td.mu, scale=td.sigma
        )
        mean_torch = truncnorm.truncnorm_mean(*td.get_torch_tensors()).numpy()
        self.assertTrue(np.allclose(mean_scipy, mean_torch))
