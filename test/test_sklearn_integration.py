import unittest

import numpy as np
import sklearn.linear_model
import sklearn.linear_model._coordinate_descent
import torch

from randalo import RandALO
from randalo import utils


class TestSklearnRandALO(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.p = 8
        self.rng = np.random.default_rng(0)
        self.X = self.rng.normal(0, 1, (self.n, self.p))
        self.beta = self.rng.normal(0, 1 / np.sqrt(self.p), (self.p))
        self.y = self.X @ self.beta + self.rng.normal(0, 1, (self.n))

    def compute_dy_hat(self, model, dy):
        model.fit(self.X, self.y)
        y_hat = model.predict(self.X)
        model.fit(self.X, self.y + dy)
        y_hat2 = model.predict(self.X)
        return y_hat2 - y_hat

    def assertRandomJacobianDirectionAlmostEqual(
        self, model, jac, epsilon=1e-6, sim_thresh=0.99, norm_rtol=1e-3
    ):
        dy = self.rng.normal(0, epsilon, (self.n,))
        dy_hat = utils.to_tensor(self.compute_dy_hat(model, dy))
        dy = utils.to_tensor(dy)

        dy_hat = dy_hat / torch.norm(dy)
        djac = jac @ dy / torch.norm(dy)
        self.assertTrue(
            torch.allclose(torch.norm(dy_hat), torch.norm(djac), rtol=norm_rtol)
        )

        if torch.norm(dy_hat) == 0 or torch.norm(djac) == 0:
            return

        sim = dy_hat / torch.norm(dy_hat) @ djac / torch.norm(djac)
        self.assertGreaterEqual(sim, sim_thresh)

    def get_randalo_jac(self, model):
        ra = RandALO.from_sklearn(model, self.X, self.y)
        return ra._jac @ torch.eye(self.n)

    def test_linear_regression(self):
        lr = sklearn.linear_model.LinearRegression(fit_intercept=False)
        lr.fit(self.X, self.y)

        # check against theoretical ground truth
        Q, _ = np.linalg.qr(self.X)
        jac = utils.to_tensor(Q @ Q.T)
        ra_jac = self.get_randalo_jac(lr)
        self.assertTrue(torch.allclose(jac, ra_jac, atol=1e-6))

        # check numerically
        self.assertRandomJacobianDirectionAlmostEqual(lr, ra_jac)

    def test_ridge(self):
        alphas = np.logspace(-1, 1, 10)

        for alpha in alphas:
            ridge = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=False)
            ridge.fit(self.X, self.y)

            # check against theoretical ground truth
            jac = utils.to_tensor(
                self.X
                @ np.linalg.solve(
                    self.X.T @ self.X + alpha * np.eye(self.p),
                    self.X.T,
                )
            )
            ra_jac = self.get_randalo_jac(ridge)
            self.assertTrue(torch.allclose(jac, ra_jac, atol=1e-6))

            # check against the wrong Jacobian
            bad_jac = utils.to_tensor(
                self.X
                @ np.linalg.solve(
                    self.X.T @ self.X + 2 * alpha * np.eye(self.p),
                    self.X.T,
                )
            )
            self.assertFalse(torch.allclose(bad_jac, ra_jac, atol=1e-6))

            # check numerically
            self.assertRandomJacobianDirectionAlmostEqual(ridge, ra_jac)

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

        # fit a lasso model for each alpha and check Jacobian
        for alpha in alphas:
            lasso = sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=False)
            lasso.fit(self.X, self.y)

            mask = lasso.coef_ != 0
            X_mask = self.X[:, mask]
            nnzs.add(np.sum(mask))

            # check against theoretical ground truth
            Q, _ = np.linalg.qr(X_mask)
            jac = utils.to_tensor(Q @ Q.T)
            ra_jac = self.get_randalo_jac(lasso)
            self.assertTrue(torch.allclose(jac, ra_jac, atol=1e-6))

            # check numerically
            self.assertRandomJacobianDirectionAlmostEqual(lasso, ra_jac, norm_rtol=0.02)

        # make sure we've actually checked different sparsity patterns
        self.assertTrue(len(nnzs) > 3)

    def test_elastic_net(self):
        # get a path of parameters
        l1_ratio = 0.5
        alphas = sklearn.linear_model._coordinate_descent._alpha_grid(
            self.X,
            self.y,
            fit_intercept=False,
            l1_ratio=l1_ratio,
            n_alphas=10,
        )

        # store the unique numbers of nonzeros over the alphas
        nnzs = set()

        # fit a lasso model for each alpha and check Jacobian
        for alpha in alphas:
            enet = sklearn.linear_model.ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False
            )
            enet.fit(self.X, self.y)

            mask = enet.coef_ != 0
            X_mask = self.X[:, mask]
            n_mask = np.sum(mask)
            nnzs.add(n_mask)

            # check against theoretical ground truth
            jac = utils.to_tensor(
                X_mask
                @ np.linalg.solve(
                    X_mask.T @ X_mask / self.n
                    + alpha * (1 - l1_ratio) * np.eye(n_mask),
                    X_mask.T / self.n,
                )
            )
            ra_jac = self.get_randalo_jac(enet)
            self.assertTrue(torch.allclose(jac, ra_jac, atol=1e-6))

            # check numerically
            self.assertRandomJacobianDirectionAlmostEqual(enet, ra_jac, norm_rtol=0.02)

        # make sure we've actually checked different sparsity patterns
        self.assertTrue(len(nnzs) > 3)


if __name__ == "__main__":
    unittest.main()
