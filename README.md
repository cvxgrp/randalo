# RandALO: fast randomized risk estimation for high-dimensional data

This repository contains a software package implementing RandALO, a fast randomized method for risk estimation of machine learning models, as described in the paper,

P. T. Nobel, D. LeJeune, E. J. Candès. RandALO: Out-of-sample risk estimation in no time flat. 2024. [arXiv:2409.09781](https://arxiv.org/abs/2409.09781).

Note: the experiments in the paper were performed in an earlier version of the codebase available in the [`paper-code`](https://github.com/cvxgrp/randalo/tree/paper-code) branch.

## Installation

In a folder run the following:

```bash
git clone git@github.com:cvxgrp/randalo.git
cd randalo

# create a new environment with Python >= 3.10 (could also use venv or similar)
conda create -n randalo python=3.12

# install requirements and randalo
pip install -r requirements.txt
```

## Usage

### Scikit-learn

The simplest way to use RandALO is with linear models from scikit-learn. See a longer demonstration in a notebook [here](examples/scikit-learn.ipynb).

```python
from torch import nn
from sklearn.linear_model import Lasso
from randalo import RandALO

X, y = ... # load data as np.ndarrays as usual

model = Lasso(1.0).fit(X, y) # fit the model
alo = RandALO.from_sklearn(model, X, y) # set up the Jacobian
mse_estimate = alo.evaluate(nn.MSELoss()) # estimate risk
```

We currently support the following models:

- `LinearRegression`
- `Ridge`
- `Lasso`
- `LassoLars`
- `ElasticNet`
- `LogisticRegression`

### Linear models with any solver

If you prefer to use other solvers for fitting your models than scikit-learn, or if you wish to extend to other models than the ones listed above, you can still use RandALO by instantiating the Jacobian yourself. You need only be careful to ensure that you scale the regularizer correctly for your problem formulation.

```python
from torch import nn
from sklearn.linear_model import Lasso
from randalo import RandALO, MSELoss, L1Regularizer, Jacobian

X, y = ... # load data as np.ndarrays as usual

model = Lasso(1.0).fit(X, y) # fit the model

# instantiate RandALO by creating a Jacobian object
loss = MSELoss()
reg = 2.0 * model.alpha * L1Regularizer() # scale the regularizer appropriately
y_hat = model.predict(X)
solution_func = lambda: model.coef_
jac = Jacobian(y, X, solution_func, loss, reg)
alo = RandALO(loss, jac, y, y_hat)

mse_estimate = alo.evaluate(nn.MSELoss()) # estimate risk
```

Please refer to our [scikit-learn integration](randalo/sklearn_integration.py) source code for more examples.
