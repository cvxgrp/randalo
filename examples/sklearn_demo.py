from sklearn.linear_model import Lasso
from randalo import RandALO

X, y, lamda = ...

lasso = Lasso(lamda)
lasso.fit(X, y)

alo = RandALO.from_sklearn(lasso, X, y)
risk_fun = lambda y, z: (y - z) ** 2
print(alo.evaluate(risk_fun, method="randalo", n_matvecs=50))
