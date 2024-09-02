import cvxpy as cp
import numpy as np

import randalo as ra

X, y = ...

beta = cp.Variable(X.shape[1])
lamda = ra.HyperParameter()
gamma = ra.HyperParameter()

loss = ra.LogisticLoss(y, X, beta)
regularizer = alpha * ra.SumSquares() + beta * ra.L1Loss(np.diff(np.eye(X.shape[1])))

prob, loss, J = ra.gen_cvxpy_loss_and_jacobian(loss, regularizer)
alpha.value = 10
beta.value = 1
prob.solve()

alo = ra.RandALO(y_hat=X @ beta.value, y=y, loss, J)
alo.compute()
