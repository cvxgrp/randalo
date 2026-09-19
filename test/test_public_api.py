import unittest

import randalo


class TestPublicAPI(unittest.TestCase):
    def test_exports(self):
        expected = {
            "HuberRegularizer",
            "HyperParameter",
            "Jacobian",
            "L1Regularizer",
            "L2Regularizer",
            "LogisticLoss",
            "Loss",
            "MSELoss",
            "NonNegativeRegularizer",
            "RandALO",
            "Regularizer",
            "SquareRegularizer",
            "Sum",
            "WeightedLoss",
            "ZeroRegularizer",
            "gen_cvxpy_jacobian",
        }
        self.assertEqual(set(randalo.__all__), expected)
        for name in expected:
            with self.subTest(name=name):
                self.assertIsNotNone(getattr(randalo, name))


if __name__ == "__main__":
    unittest.main()
