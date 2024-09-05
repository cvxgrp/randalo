import unittest

import linops as lo
import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

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
        prob = reductions.transform_model_to_cvxpy(self.loss, self.regularizer, self.X, y, b)
        self.layer = CvxpyLayer(prob, parameters=[y], variables=[b])

        y_torch = torch.tensor(self.y, requires_grad=True)
        J = reductions.Jacobian(self.y, self.X, lambda: b.value, self.loss, self.regularizer)

        y.value = self.y
        prob.solve()
        y_hat_torch = torch.from_numpy(self.X) @ self.layer(y_torch)[0]
        
        z = self.rng.standard_normal(self.n)
        Jz = J @ z
        (y_hat_torch @ torch.from_numpy(z)).backward()
        assert torch.allclose(Jz, y_torch.grad.to(torch.float32), atol=1e-4, rtol=1e-2)



if __name__ == '__main__':
    unittest.main()
