import sklearn.linear_model import Lasso
import randalo as ra


X, y, lamda = ...

lasso = Lasso(lamda)
lasso.fit(X, y)
y_hat = lasso.predict(X)

loss, J = ra.gen_loss_and_jacobian(lasso)

alo_cf_hessian = ra.RandALO(y_hat=y_hat, y=y, loss, J)

