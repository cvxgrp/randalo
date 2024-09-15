import unittest

import numpy as np
import torch

from randalo import utils


class TestUtils(unittest.TestCase):
    def test_to_tensor(self):
        x = [1, 2, 3]
        x_np = np.array(x)
        self.assertIs(x_np.dtype, np.dtype("int64"))
        x_torch = torch.tensor(x)
        self.assertIs(x_torch.dtype, torch.int64)
        x_torch = x_torch.to(torch.float32)

        x_to_tensor = utils.to_tensor(x)
        self.assertTrue(torch.allclose(x_torch, x_to_tensor))
        self.assertIs(x_to_tensor.dtype, torch.float32)

        x_np_to_tensor = utils.to_tensor(x_np)
        self.assertTrue(torch.allclose(x_torch, x_np_to_tensor))
        self.assertIs(x_np_to_tensor.dtype, torch.float32)

        x_torch_to_tensor = utils.to_tensor(x_torch)
        self.assertTrue(torch.allclose(x_torch, x_torch_to_tensor))
        self.assertIs(x_torch_to_tensor.dtype, torch.float32)

        with self.assertRaises(ValueError):
            utils.to_tensor(1)

    def test_compute_derivatives(self):
        n = 100

        def loss_fun(y, z):
            return torch.sum((y - z) ** 4)

        rng = torch.Generator().manual_seed(0)
        y = torch.randn(n, generator=rng)
        z = torch.randn(n, generator=rng)

        y_, z_, dloss_dz, d2loss_dboth, d2loss_dz2 = utils.compute_derivatives(
            loss_fun, y, z
        )
        self.assertTrue(torch.allclose(y, y_))
        self.assertTrue(torch.allclose(z, z_))
        self.assertTrue(torch.allclose(4 * (z - y) ** 3, dloss_dz))
        self.assertTrue(torch.allclose(-12 * (y - z) ** 2, d2loss_dboth))
        self.assertTrue(torch.allclose(12 * (y - z) ** 2, d2loss_dz2))

    def test_unsqueeze_scalar_like(self):
        x = 1.5
        arr1 = torch.ones(10, dtype=torch.float32)
        arr2 = torch.ones(10, 10, dtype=torch.float64)
        arr3 = torch.ones(10, 10, 10, dtype=torch.int32)

        x1 = utils.unsqueeze_scalar_like(x, arr1)
        self.assertEqual(x1.shape, (1,))
        self.assertIs(x1.dtype, torch.float32)
        self.assertEqual(x1.item(), x)

        x2 = utils.unsqueeze_scalar_like(x, arr2)
        self.assertEqual(x2.shape, (1, 1))
        self.assertIs(x2.dtype, torch.float64)
        self.assertEqual(x2.item(), x)

        x3 = utils.unsqueeze_scalar_like(x, arr3)
        self.assertEqual(x3.shape, (1, 1, 1))
        self.assertIs(x3.dtype, torch.int32)
        self.assertNotEqual(x3.item(), x)
        self.assertEqual(x3.item(), np.floor(x))

    def test_create_mixing_matrix(self):
        m = 5
        subsets = [[0, 1, 2], [2, 3], [4]]
        M0 = torch.tensor(
            [[1 / 3, 0, 0], [1 / 3, 0, 0], [1 / 3, 1 / 2, 0], [0, 1 / 2, 0], [0, 0, 1]]
        )
        M = utils.create_mixing_matrix(m, subsets)
        self.assertTrue(torch.allclose(M0, M))

    def test_robust_y_intercept(self):
        n = 100
        rng = torch.Generator().manual_seed(0)
        x = torch.randn(n // 2, generator=rng)
        y0 = np.pi
        z = torch.randn(n // 2, generator=rng)
        y = torch.concatenate([y0 + x + z, y0 + x - z])
        x = torch.concatenate([x, x])

        y0_hat = utils.robust_y_intercept(x, y)
        self.assertAlmostEqual(y0, y0_hat, places=4)


if __name__ == "__main__":
    unittest.main()
