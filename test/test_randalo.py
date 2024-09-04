import unittest

import linops as lo
import numpy as np
import torch

from randalo import RandALO, modeling_layer as ml


class TestRandALO(unittest.TestCase):

    def test_diagonal_jac(self):

        n = 10
        rng = torch.Generator().manual_seed(0)
        diag = torch.rand(n, generator=rng)
        jac = lo.DiagonalOperator(diag)
        loss = ml.MSELoss()
        y = torch.rand(n, generator=rng)
        y_hat = torch.rand(n, generator=rng)

        ra = RandALO(loss=loss, jac=jac, y=y, y_hat=y_hat)
        risk = ra.evaluate(loss, n_matvecs=3)
        self.assertEqual(ra._n_matvecs, 3)
        self.assertEqual(ra._normalized_diag_jac_estims.shape, (n, 3))
        self.assertTrue(torch.allclose(ra._normalized_diag_jac_estims, diag[:, None]))
        self.assertTrue(torch.allclose(ra._normalized_diag_jac_stds, torch.zeros(n, 1)))

        risk_bks = ra.evaluate_bks(loss, n_matvecs=3)
        risk_alo = ra.evaluate_alo(loss)
        self.assertAlmostEqual(risk, risk_bks)
        self.assertAlmostEqual(risk, risk_alo)

        self.assertTrue(
            torch.allclose(ra._y_tilde_exact, y_hat + diag / (1 - diag) * (y_hat - y))
        )
        self.assertAlmostEqual(risk, (loss(y, y_hat) / (1 - diag) ** 2).mean().item())
