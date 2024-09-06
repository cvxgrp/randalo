import unittest

import linops as lo
import numpy as np
import sklearn.linear_model
import sklearn.linear_model._coordinate_descent
import torch

from randalo import RandALO, SklearnRandALO
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

    def test_diagonal_jac(self):
        ra = RandALO(
            loss=self.loss, jac=self.jac, y=self.y, y_hat=self.y_hat, rng=self.rng
        )
        risk = ra.evaluate(self.loss, n_matvecs=3)
        self.assertEqual(ra._n_matvecs, 3)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (self.n, 3))
        self.assertTrue(
            torch.allclose(ra._normalized_diag_jac_estims, self.diag[:, None])
        )
        self.assertTrue(
            torch.allclose(ra._normalized_diag_jac_stds, torch.zeros(self.n, 1))
        )

        risk_bks = ra.evaluate_bks(self.loss, n_matvecs=3)
        risk_alo = ra.evaluate_alo(self.loss)
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
        risk = ra.evaluate(self.loss, n_matvecs=200_000)
        self.assertFalse(torch.allclose(ra._normalized_diag_jac_estims, diag[:, None]))

        risk_bks = ra.evaluate_bks(self.loss, n_matvecs=200_000)
        risk_alo = ra.evaluate_alo(self.loss)
        self.assertAlmostEqual(risk, risk_bks, places=3)
        self.assertAlmostEqual(risk, risk_alo, places=3)
        self.assertAlmostEqual(
            risk_alo,
            torch.mean((self.y - self.y_hat) ** 2 / (1 - diag) ** 2).item(),
        )

    def test_more_matvecs(self):
        ra = RandALO(
            loss=self.loss, jac=self.jac, y=self.y, y_hat=self.y_hat, rng=self.rng
        )
        ra.evaluate(self.loss, n_matvecs=3)
        self.assertEqual(ra._n_matvecs, 3)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (self.n, 3))

        ra.evaluate_bks(self.loss, n_matvecs=5)
        self.assertEqual(ra._n_matvecs, 5)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (self.n, 5))

        # test that the last matevecs are not used if we request fewer than we have
        ra._normalized_diag_jac_estims[:, -1] = torch.nan
        risk1 = ra.evaluate(self.loss, n_matvecs=2)
        self.assertEqual(ra._n_matvecs, 5)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (self.n, 5))
        self.assertFalse(np.isnan(risk1))

        risk2 = ra.evaluate_bks(self.loss, n_matvecs=4)
        self.assertEqual(ra._n_matvecs, 5)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (self.n, 5))
        self.assertFalse(np.isnan(risk2))

        # but if we do use them, we should get nan
        risk3 = ra.evaluate_bks(self.loss, n_matvecs=6)
        self.assertEqual(ra._n_matvecs, 6)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (self.n, 6))
        self.assertTrue(np.isnan(risk3))

    def test_uniform_map_estimates(self):
        ra = RandALO(
            loss=self.loss, jac=self.jac, y=self.y, y_hat=self.y_hat, rng=self.rng
        )
        ra.evaluate(self.loss, n_matvecs=3)

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


class TestSklearnRandALO(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.p = 8
        self.rng = np.random.default_rng(0)
        self.X = self.rng.normal(0, 1, (self.n, self.p))
        self.beta = self.rng.normal(0, 1 / np.sqrt(self.p), (self.p))
        self.y = self.X @ self.beta + self.rng.normal(0, 1, (self.n))

    def test_linear_regression(self):
        lr = sklearn.linear_model.LinearRegression(fit_intercept=False)
        lr.fit(self.X, self.y)

        Q, _ = np.linalg.qr(self.X)
        jac = utils.to_tensor(Q @ Q.T)

        ra = SklearnRandALO(lr, self.X, self.y)
        ra_jac = ra._jac @ torch.eye(self.n)

        self.assertTrue(torch.allclose(jac, ra_jac, atol=1e-6))

    def test_lasso(self):
        # get a path of parameters
        alphas = sklearn.linear_model._coordinate_descent._alpha_grid(
            self.X,
            self.y,
            fit_intercept=False,
            l1_ratio=1.0,
            n_alphas=10,
        )

        # store the unique numbers of nonzeros over the alphas
        nnzs = set()

        # fit a lasso model for each alpha and check
        for alpha in alphas:
            lasso = sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=False)
            lasso.fit(self.X, self.y)

            mask = lasso.coef_ != 0
            X_mask = self.X[:, mask]
            nnzs.add(np.sum(mask))

            Q, _ = np.linalg.qr(X_mask)
            jac = utils.to_tensor(Q @ Q.T)

            ra = SklearnRandALO(lasso, self.X, self.y)
            ra_jac = ra._jac @ torch.eye(self.n)

            self.assertTrue(torch.allclose(jac, ra_jac, atol=1e-6))

            # numerically check the derivative in a random direction
            y_hat = lasso.predict(self.X)
            dy = self.rng.normal(0, 1e-6, (self.n,))
            lasso.fit(self.X, self.y + dy)
            y_hat2 = lasso.predict(self.X)

            dy = utils.to_tensor(dy)
            dy_hat = utils.to_tensor(y_hat2 - y_hat)

            self.assertTrue(torch.allclose(dy_hat, ra_jac @ dy, atol=1e-6))

        # make sure we've actually checked different sparsity patterns
        self.assertTrue(len(nnzs) > 3)


if __name__ == "__main__":
    unittest.main()
