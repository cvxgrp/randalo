import cvxpy as cp
import numpy as np

import randalo as ra

X, y = ...

beta = cp.Variable(X.shape[1])
lamda = ra.HyperParameter()
gamma = ra.HyperParameter()

loss = ra.LogisticLoss()
regularizer = lamda * ra.SquareRegularizer() + gamma * ra.L1Regularizer(np.diff(np.eye(X.shape[1])))

prob, J = ra.gen_cvxpy_jacobian(loss, regularizer, X, beta, y)
lamda.value = 10
gamma.value = 1
prob.solve()

alo = ra.RandALO(y_hat=X @ beta.value, y=y, loss=loss, J=J).evaluate(loss)

