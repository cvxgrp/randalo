import unittest

import numpy as np
import scipy
import scipy.special
import sklearn.linear_model
import sklearn.linear_model._coordinate_descent
import torch

from randalo import RandALO
from randalo import modeling_layer as ml
from randalo import utils


class TestSklearnRandALO(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.n = 10
        self.p = 8
        self.rng = np.random.default_rng(0)
        self.X = self.rng.normal(0, 1, (self.n, self.p))
        self.beta = self.rng.normal(0, 1 / np.sqrt(self.p), (self.p))
        self.y = self.X @ self.beta + self.rng.normal(0, 1, (self.n))
        self.y_bin = (
            self.rng.uniform(0, 1, (self.n,))
            < scipy.special.expit(4 * self.X @ self.beta)
        ).astype(int)

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

    def get_randalo_jac(self, model, y=None):
        if y is None:
            y = self.y
        ra = RandALO.from_sklearn(model, self.X, y)
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
        # get a path of parameters and shrink a bit so we aren't at the max
        alphas = (
            sklearn.linear_model._coordinate_descent._alpha_grid(
                self.X,
                self.y,
                fit_intercept=False,
                l1_ratio=1.0,
                n_alphas=10,
            )
            * 0.99
        )

        # store the unique numbers of nonzeros over the alphas
        nnzs = set()

        # fit a lasso model for each alpha and check Jacobian
        for alpha in alphas:
            lasso = sklearn.linear_model.Lasso(
                alpha=alpha, fit_intercept=False, tol=1e-8
            )
            lasso.fit(self.X, self.y)
            lasso_lars = sklearn.linear_model.LassoLars(
                alpha=alpha, fit_intercept=False, max_iter=10000
            )
            lasso_lars.fit(self.X, self.y)

            mask = lasso.coef_ != 0
            X_mask = self.X[:, mask]
            nnzs.add(np.sum(mask))

            # check against theoretical ground truth
            Q, _ = np.linalg.qr(X_mask)
            jac = utils.to_tensor(Q @ Q.T)
            ra_jac = self.get_randalo_jac(lasso)
            ra_jac_lars = self.get_randalo_jac(lasso_lars)
            self.assertTrue(torch.allclose(jac, ra_jac, atol=1e-6))
            self.assertTrue(torch.allclose(jac, ra_jac_lars, atol=1e-6))

            # check numerically
            self.assertRandomJacobianDirectionAlmostEqual(
                lasso, ra_jac, sim_thresh=0.999, norm_rtol=1e-3
            )
            self.assertRandomJacobianDirectionAlmostEqual(
                lasso_lars, ra_jac, sim_thresh=0.9999, norm_rtol=1e-2
            )

        # make sure we've actually checked different sparsity patterns
        self.assertTrue(len(nnzs) > 3)

    def test_elastic_net(self):
        # get a path of parameters and shrink a bit
        l1_ratio = 0.5
        alphas = (
            sklearn.linear_model._coordinate_descent._alpha_grid(
                self.X,
                self.y,
                fit_intercept=False,
                l1_ratio=l1_ratio,
                n_alphas=10,
            )
            * 0.99
        )

        # store the unique numbers of nonzeros over the alphas
        nnzs = set()

        # fit a lasso model for each alpha and check Jacobian
        for alpha in alphas:
            enet = sklearn.linear_model.ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, tol=1e-8
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

    def test_logistic(self):
        param_dicts = [
            {"penalty": None, "eye_scale": 0.0},
            {"penalty": "l1", "C": 1.0, "eye_scale": 0.0},
            {"penalty": "l2", "C": 1.0, "eye_scale": 1.0},
            {"penalty": "elasticnet", "C": 1.0, "l1_ratio": 0.5, "eye_scale": 0.5},
        ]
        lr = sklearn.linear_model.LogisticRegression(
            tol=1e-5, solver="saga", max_iter=100000, fit_intercept=False
        )

        y = utils.to_tensor(self.y_bin * 2 - 1)
        y_labels = np.array(["a", "b"])[self.y_bin]
        nnzs = set()

        for param_dict in param_dicts:
            eye_scale = param_dict.pop("eye_scale")
            lr.set_params(**param_dict)
            lr.fit(self.X, y_labels)

            mask = lr.coef_[0, :] != 0
            X_mask = self.X[:, mask]
            n_mask = np.sum(mask)
            nnzs.add(n_mask)

            y_hat = utils.to_tensor(lr.decision_function(self.X))
            ds = utils.compute_derivatives(ml.LogisticLoss(), y, y_hat)
            d2loss_dboth = ds.d2loss_dboth.numpy()
            d2loss_dy_hat2 = ds.d2loss_dy_hat2.numpy()

            jac = utils.to_tensor(
                -X_mask
                @ np.linalg.solve(
                    X_mask.T @ (d2loss_dy_hat2[:, None] * X_mask)
                    + eye_scale / self.n * np.eye(n_mask),
                    X_mask.T * d2loss_dboth[None, :],
                )
            )
            ra_jac = self.get_randalo_jac(lr, y=y_labels)
            atol = 1e-2 if param_dict["penalty"] is None else 1e-6
            self.assertTrue(torch.allclose(jac, ra_jac, atol=atol))

            # check that wrong y = wrong Jacobian
            ds_bad = utils.compute_derivatives(ml.LogisticLoss(), -y, y_hat)
            d2loss_dboth_bad = ds_bad.d2loss_dboth.numpy()
            d2loss_dy_hat2_bad = ds_bad.d2loss_dy_hat2.numpy()

            jac_bad = utils.to_tensor(
                -X_mask
                @ np.linalg.solve(
                    X_mask.T @ (d2loss_dy_hat2_bad[:, None] * X_mask)
                    + eye_scale / self.n * np.eye(n_mask),
                    X_mask.T * d2loss_dboth_bad[None, :],
                )
            )
            self.assertFalse(torch.allclose(jac_bad, ra_jac, atol=atol))


if __name__ == "__main__":
    unittest.main()
