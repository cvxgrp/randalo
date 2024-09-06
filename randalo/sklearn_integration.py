import numpy as np
import torch

import sklearn.base
import sklearn.linear_model
import sklearn.utils.validation

from . import modeling_layer as ml
from . import reductions


def map_sklearn_to_loss_jac_y_hat(
    model: sklearn.base.BaseEstimator = None,
    X: torch.Tensor | np.ndarray = None,
    y: torch.Tensor | np.ndarray = None,
):
    sklearn.utils.validation.check_is_fitted(model)
    n = X.shape[0]

    match model:
        case sklearn.linear_model.LinearRegression():
            loss = ml.MSELoss()
            # TODO create trivial regularizer
            reg = 0.0 * ml.SquareRegularizer()
            y_hat = model.predict(X)

        case sklearn.linear_model.Ridge():
            loss = ml.MSELoss()
            reg = model.alpha / n * ml.SquareRegularizer()
            y_hat = model.predict(X)

        case sklearn.linear_model.Lasso():
            loss = ml.MSELoss()
            reg = 2.0 * model.alpha * ml.L1Regularizer()
            y_hat = model.predict(X)

        case sklearn.linear_model.ElasticNet():
            loss = ml.MSELoss()
            reg = model.alpha * (
                2.0 * model.l1_ratio * ml.L1Regularizer()
                + (1 - model.l1_ratio) * ml.SquareRegularizer()
            )
            y_hat = model.predict(X)

    jac = reductions.Jacobian(y, X, lambda: model.coef_, loss, reg, inverse_method=None)

    return loss, jac, y_hat
