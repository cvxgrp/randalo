import cvxpy as cp
import numpy as np

import randalo as ra


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 20))
    beta_true = np.zeros(20)
    beta_true[:3] = 1.0
    y = X @ beta_true + 0.1 * rng.standard_normal(100)

    beta = cp.Variable(X.shape[1])
    regularization = ra.HyperParameter()
    loss = ra.MSELoss()
    regularizer = regularization * ra.SquareRegularizer()

    problem, jac = ra.gen_cvxpy_jacobian(
        loss, regularizer, X, beta, y
    )
    regularization.value = 0.1
    problem.solve()

    alo = ra.RandALO(
        y_hat=X @ beta.value,
        y=y,
        loss=loss,
        jac=jac,
    )
    return alo.evaluate(loss, n_matvecs=20, subsets=10)


if __name__ == "__main__":
    print(main())
