import unittest

import cvxpy as cp
import numpy as np
import scipy.sparse
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
        _, generated_jacobian = reductions.gen_cvxpy_jacobian(
            self.loss,
            self.regularizer,
            self.X,
            b,
            self.y,
            inversion_method="minres",
        )
        self.assertEqual(generated_jacobian.inverse_method, "minres")

        solve_options = {
            "solver": "CLARABEL",
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
            "tol_feas": 1e-10,
        }
        y.value = self.y
        prob.solve(**solve_options)
        direction = self.rng.standard_normal(self.n)
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
        y.value = self.y
        prob.solve(**solve_options)
        for inverse_method in (None, "cholesky", "minres"):
            with self.subTest(inverse_method=inverse_method):
                jacobian = reductions.Jacobian(
                    self.y,
                    self.X,
                    lambda: b.value,
                    self.loss,
                    self.regularizer,
                    inverse_method=inverse_method,
                )
                actual = jacobian @ direction
                self.assertTrue(
                    torch.allclose(actual, expected, atol=1e-3, rtol=1e-2)
                )

        with self.assertRaisesRegex(ValueError, "inverse_method"):
            reductions.Jacobian(
                self.y,
                self.X,
                lambda: b.value,
                self.loss,
                self.regularizer,
                inverse_method="unknown",
            )

        with self.assertRaisesRegex(ValueError, "dense design matrices"):
            reductions.Jacobian(
                self.y,
                scipy.sparse.csr_matrix(self.X),
                lambda: b.value,
                self.loss,
                self.regularizer,
                inverse_method="minres",
            )

    def test_constrained_jacobian(self):
        rng = np.random.default_rng(13)
        X = rng.standard_normal((12, 4))
        beta = np.array([1.0, 1.0, 0.4, -0.3])
        y_value = X @ beta + 0.05 * rng.standard_normal(12)
        difference = np.array([[1.0, -1.0, 0.0, 0.0]])
        regularizer = (
            0.1 * ml.L1Regularizer(linear=difference)
            + 0.01 * ml.SquareRegularizer()
        )

        variable = cp.Variable(4)
        y = cp.Parameter(12)
        problem = reductions.transform_model_to_cvxpy(
            ml.MSELoss(), regularizer, X, y, variable
        )
        solve_options = {
            "solver": "CLARABEL",
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
            "tol_feas": 1e-10,
        }
        y.value = y_value
        problem.solve(**solve_options)
        self.assertLess(np.linalg.norm(difference @ variable.value), 1e-6)

        direction = rng.standard_normal(12)
        epsilon = 1e-4
        predictions = []
        for sign in (-1, 1):
            y.value = y_value + sign * epsilon * direction
            problem.solve(warm_start=True, **solve_options)
            predictions.append(X @ variable.value)
        expected = torch.as_tensor(
            (predictions[1] - predictions[0]) / (2 * epsilon),
            dtype=torch.float32,
        )

        y.value = y_value
        problem.solve(**solve_options)
        cases = [
            (X, None),
            (X, "cholesky"),
            (X, "minres"),
            (scipy.sparse.csr_matrix(X), None),
        ]
        for design, inverse_method in cases:
            with self.subTest(
                sparse=scipy.sparse.issparse(design),
                inverse_method=inverse_method,
            ):
                jacobian = reductions.Jacobian(
                    y_value,
                    design,
                    lambda: variable.value,
                    ml.MSELoss(),
                    regularizer,
                    inverse_method=inverse_method,
                )
                actual = jacobian @ direction
                self.assertTrue(
                    torch.allclose(actual, expected, atol=1e-4, rtol=1e-3)
                )

    def test_regularizer_hessians(self):
        beta = torch.tensor([1.2, -0.7, 0.3], dtype=torch.float64)
        cases = [
            (None, lambda value: torch.linalg.norm(value)),
            ([0, 2], lambda value: torch.linalg.norm(value[[0, 2]])),
            (
                np.array([[1.0, -1.0, 0.0], [0.0, 1.0, 2.0]]),
                lambda value: torch.linalg.norm(
                    torch.tensor(
                        [[1.0, -1.0, 0.0], [0.0, 1.0, 2.0]],
                        dtype=value.dtype,
                    )
                    @ value
                ),
            ),
        ]
        for linear, function in cases:
            with self.subTest(linear=linear):
                regularizer = 1.7 * ml.L2Regularizer(linear=linear)
                _, actual, _ = regularizer.get_constraint_hessian_mask(beta)
                expected = torch.autograd.functional.hessian(
                    lambda value: 1.7 * function(value), beta
                )
                self.assertTrue(torch.allclose(actual, expected, atol=1e-10))

        huber = 0.5 * ml.HuberRegularizer()
        _, hessian, _ = huber.get_constraint_hessian_mask(beta)
        self.assertTrue(
            torch.equal(hessian, torch.diag(torch.tensor([0.0, 1.0, 1.0])))
        )


if __name__ == "__main__":
    unittest.main()
