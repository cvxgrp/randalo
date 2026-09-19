import unittest

import cvxpy as cp
import numpy as np
import torch

from randalo import modeling_layer as ml, reductions


class TestReductions(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.rng = np.random.default_rng(0x219A)
        self.loss = ml.MSELoss()
        self.regularizer = 0.01 * (0.5 * ml.SquareRegularizer() + ml.L1Regularizer())
        self.X = self.rng.standard_normal((self.n, 3 * self.n))
        self.y = self.X[:, 0] + 0.01 * self.rng.standard_normal(self.n)

    def test_transform_to_cvxpy_and_test_jacobian(self):
        b = cp.Variable(3 * self.n)
        y = cp.Parameter(self.n)
        prob = reductions.transform_model_to_cvxpy(
            self.loss,
            self.regularizer,
            self.X,
            y,
            b,
        )
        jacobian = reductions.Jacobian(
            self.y,
            self.X,
            lambda: b.value,
            self.loss,
            self.regularizer,
            inverse_method="cholesky",
        )
        _, generated_jacobian = reductions.gen_cvxpy_jacobian(
            self.loss,
            self.regularizer,
            self.X,
            b,
            self.y,
            inversion_method="cholesky",
        )
        self.assertEqual(generated_jacobian.inverse_method, "cholesky")

        solve_options = {
            "solver": "CLARABEL",
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
            "tol_feas": 1e-10,
        }
        y.value = self.y
        prob.solve(**solve_options)
        direction = self.rng.standard_normal(self.n)
        actual = jacobian @ direction

        epsilon = 1e-4
        predictions = []
        for sign in (-1, 1):
            y.value = self.y + sign * epsilon * direction
            prob.solve(warm_start=True, **solve_options)
            predictions.append(self.X @ b.value)
        expected = torch.as_tensor(
            (predictions[1] - predictions[0]) / (2 * epsilon),
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-3, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
