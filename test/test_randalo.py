import unittest

import linops as lo
import numpy as np
import torch

from randalo import RandALO
from randalo import modeling_layer as ml
from randalo import truncnorm
from randalo import utils


class TestRandALO(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.rng = torch.Generator().manual_seed(0)
        self.diag = torch.rand(self.n, generator=self.rng)
        self.jac = lo.DiagonalOperator(self.diag)
        self.loss = ml.MSELoss()
        self.y = torch.rand(self.n, generator=self.rng)
        self.y_hat = torch.rand(self.n, generator=self.rng)

    def risk_fun(self, y, z):
        return torch.mean((y - z) ** 2).item()

    def test_diagonal_jac(self):
        ra = RandALO(
            loss=self.loss, jac=self.jac, y=self.y, y_hat=self.y_hat, rng=self.rng
        )
        risk = ra.evaluate(self.risk_fun, n_matvecs=3)
        self.assertEqual(ra._n_matvecs, 3)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (self.n, 3))
        self.assertTrue(
            torch.allclose(ra._normalized_diag_jac_estims, self.diag[:, None])
        )
        self.assertTrue(
            torch.allclose(ra._normalized_diag_jac_stds, torch.zeros(self.n, 1))
        )

        risk_bks = ra.evaluate_bks(self.risk_fun, n_matvecs=3)
        risk_alo = ra.evaluate_alo(self.risk_fun)
        self.assertAlmostEqual(risk, risk_bks)
        self.assertAlmostEqual(risk, risk_alo)

        self.assertTrue(
            torch.allclose(
                ra._y_tilde_exact,
                self.y_hat + self.diag / (1 - self.diag) * (self.y_hat - self.y),
            )
        )
        self.assertAlmostEqual(
            risk,
            torch.mean((self.y - self.y_hat) ** 2 / (1 - self.diag) ** 2).item(),
        )

    def test_psd_jac(self):
        X = torch.rand(self.n, self.n, generator=self.rng)
        J = X @ torch.linalg.solve(X.T @ X + torch.eye(self.n), X.T)
        diag = torch.diag(J)
        jac = lo.MatrixOperator(J)

        ra = RandALO(loss=self.loss, jac=jac, y=self.y, y_hat=self.y_hat, rng=self.rng)
        risk = ra.evaluate(self.risk_fun, n_matvecs=200_000)
        self.assertFalse(torch.allclose(ra._normalized_diag_jac_estims, diag[:, None]))

        risk_bks = ra.evaluate_bks(self.risk_fun, n_matvecs=200_000)
        risk_alo = ra.evaluate_alo(self.risk_fun)
        self.assertAlmostEqual(risk, risk_bks, places=3)
        self.assertAlmostEqual(risk, risk_alo, places=3)
        self.assertAlmostEqual(
            risk_alo,
            torch.mean((self.y - self.y_hat) ** 2 / (1 - diag) ** 2).item(),
        )

    def test_logistic(self):
        loss = ml.LogisticLoss()
        X = torch.rand(self.n, self.n, generator=self.rng)
        y = torch.randint(0, 2, (self.n,), generator=self.rng).float() * 2 - 1
        y_hat = torch.randn(self.n, generator=self.rng)

        ds = utils.compute_derivatives(loss, y, y_hat)
        dloss_dy_hat = ds.dloss_dy_hat
        d2loss_dboth = ds.d2loss_dboth
        d2loss_dy_hat2 = ds.d2loss_dy_hat2

        jac = -lo.MatrixOperator(
            X
            @ torch.linalg.solve(
                X.T @ (d2loss_dy_hat2[:, None] * X) + torch.eye(self.n),
                X.T * d2loss_dboth[None, :],
            )
        )
        jac_tilde = X @ torch.linalg.solve(
            X.T @ (d2loss_dy_hat2[:, None] * X) + torch.eye(self.n),
            X.T * d2loss_dy_hat2[None, :],
        )

        ra = RandALO(loss=loss, jac=jac, y=y, y_hat=y_hat, rng=self.rng)

        def risk_fun(y, z):
            return torch.mean(torch.lt(y * z, 0).float()).item()

        jac_tilde_diag = torch.diag(jac_tilde)
        risk = ra.evaluate_alo(risk_fun)
        risk_bks = ra.evaluate_bks(risk_fun, n_matvecs=200)
        self.assertAlmostEqual(risk, risk_bks)

        y_tilde_exact = y_hat + dloss_dy_hat / d2loss_dy_hat2 * jac_tilde_diag / (
            1 - jac_tilde_diag
        )
        self.assertTrue(torch.allclose(ra._y_tilde_exact, y_tilde_exact))
        self.assertAlmostEqual(risk, risk_fun(y, y_tilde_exact))

    def test_more_matvecs(self):
        ra = RandALO(
            loss=self.loss, jac=self.jac, y=self.y, y_hat=self.y_hat, rng=self.rng
        )
        ra.evaluate(self.risk_fun, n_matvecs=3)
        self.assertEqual(ra._n_matvecs, 3)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (self.n, 3))

        ra.evaluate_bks(self.risk_fun, n_matvecs=5)
        self.assertEqual(ra._n_matvecs, 5)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (self.n, 5))

        # test that the last matevecs are not used if we request fewer than we have
        ra._normalized_diag_jac_estims[:, -1] = torch.nan
        risk1 = ra.evaluate(self.risk_fun, n_matvecs=2)
        self.assertEqual(ra._n_matvecs, 5)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (self.n, 5))
        self.assertFalse(np.isnan(risk1))

        risk2 = ra.evaluate_bks(self.risk_fun, n_matvecs=4)
        self.assertEqual(ra._n_matvecs, 5)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (self.n, 5))
        self.assertFalse(np.isnan(risk2))

        # but if we do use them, we should get nan
        risk3 = ra.evaluate_bks(self.risk_fun, n_matvecs=6)
        self.assertEqual(ra._n_matvecs, 6)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (self.n, 6))
        self.assertTrue(np.isnan(risk3))

    def test_uniform_map_estimates(self):
        ra = RandALO(
            loss=self.loss, jac=self.jac, y=self.y, y_hat=self.y_hat, rng=self.rng
        )
        ra.evaluate(self.risk_fun, n_matvecs=3)

        mus = ra._normalized_diag_jac_estims.mean(dim=1, keepdim=True)
        stds = ra._normalized_diag_jac_stds[:, None]
        ms = torch.arange(1, 4, dtype=torch.float32)[None, :]
        sigmas = stds / torch.sqrt(ms)
        uniform_map_estimates = ra._uniform_map_estimates(mus, ms)
        truncnorm_means = truncnorm.truncnorm_mean(
            mus, sigmas, torch.tensor([[0.0]]), torch.tensor([[1.0]])
        )
        self.assertTrue(torch.allclose(uniform_map_estimates, truncnorm_means))

        # add another dimension
        mus = mus[..., None]
        stds = stds[..., None]
        ms = ms[..., None]
        sigmas = stds / torch.sqrt(ms)
        uniform_map_estimates = ra._uniform_map_estimates(mus, ms)
        truncnorm_means = truncnorm.truncnorm_mean(
            mus, sigmas, torch.tensor([[[0.0]]]), torch.tensor([[[1.0]]])
        )
        self.assertTrue(torch.allclose(uniform_map_estimates, truncnorm_means))


if __name__ == "__main__":
    unittest.main()
