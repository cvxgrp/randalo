# RandALO: fast randomized risk estimation for high-dimensional data

This repository contains a software package implementing RandALO, a fast randomized method for risk estimation of machine learning models, as described in the paper,

P. T. Nobel, D. LeJeune, E. J. Candès. RandALO: Out-of-sample risk estimation in no time flat. 2024.

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

The simplest way to use RandALO is with linear models from `scikit-learn`. See a longer demonstration in a notebook [here](examples/scikit-learn.ipynb).

```python
from torch import nn
from sklearn.linear_model import Lasso
from randalo import RandALO

X, y = ... # load data as np.ndarrays as usual

model = Lasso(1.0).fit(X, y)
alo = RandALO.from_sklearn(model, X, y)
mse_estimate = alo.evaluate(nn.MSELoss())
```

We currently support the following models:

- `LinearRegression`
- `Ridge`
- `Lasso`
- `LassoLars`
- `ElasticNet`
- `LogisticRegression`
