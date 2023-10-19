import unittest

from scipy import stats
import torch

from alogcv import truncnorm


class TestTruncNormMean(unittest.TestCase):
    """Test truncnorm_mean function."""

    def test_truncnorm_mean(self):
        # Sample data
        mu = torch.tensor([0.0, 1.0, 2.0])
        sigma = torch.tensor([1.0, 0.5, 1.5])
        a = torch.tensor([-1.0, 0.5, 1.0])
        b = torch.tensor([1.0, 2.0, 3.0])

        # Calculate mean using the torch implementation
        result = truncnorm.truncnorm_mean(mu, sigma, a, b)

        # Calculate expected values using scipy.stats.truncnorm.mean
        expected = torch.tensor(
            [
                stats.truncnorm.mean(
                    a=(a[i] - mu[i]) / sigma[i],
                    b=(b[i] - mu[i]) / sigma[i],
                    loc=mu[i],
                    scale=sigma[i],
                )
                for i in range(len(mu))
            ],
            dtype=torch.float32,
        )

        # Check if the results are approximately equal within a small tolerance
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
