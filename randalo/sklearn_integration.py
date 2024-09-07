import numpy as np
import torch

import sklearn.base
import sklearn.linear_model
import sklearn.utils.validation

from . import modeling_layer as ml
from . import reductions


def map_sklearn(
    model: sklearn.base.BaseEstimator = None,
    X: torch.Tensor | np.ndarray = None,
    y: torch.Tensor | np.ndarray | list = None,
):
    sklearn.utils.validation.check_is_fitted(model)
    n = X.shape[0]

    def solution_func():
        return model.coef_

    match model:
        case sklearn.linear_model.LinearRegression():
            loss = ml.MSELoss()
            # TODO create trivial regularizer
            reg = 0 * ml.SquareRegularizer()
            y_hat = model.predict(X)

        case sklearn.linear_model.Ridge():
            loss = ml.MSELoss()
            reg = model.alpha / n * ml.SquareRegularizer()
            y_hat = model.predict(X)

        case sklearn.linear_model.Lasso() | sklearn.linear_model.LassoLars():
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

        case sklearn.linear_model.LogisticRegression():
            # TODO: implement sample weight
            if len(model.classes_) > 2:
                raise ValueError("Only binary classification is supported.")

            loss = ml.LogisticLoss()
            y = ((model.classes_[None, :] == y[:, None]) * np.array([[-1, 1]])).sum(1)
            match model.penalty:
                case None:
                    # TODO: create trivial regularizer
                    reg = 0 * ml.SquareRegularizer()
                case "l1":
                    reg = ml.L1Regularizer()
                case "l2":
                    reg = 0.5 * ml.SquareRegularizer()
                case "elasticnet":
                    reg = (
                        model.l1_ratio * ml.L1Regularizer()
                        + 0.5 * (1 - model.l1_ratio) * ml.SquareRegularizer()
                    )
            reg = reg * 1 / model.C / n
            y_hat = model.decision_function(X)

            def solution_func():
                return model.coef_[0, :]

    jac = reductions.Jacobian(y, X, solution_func, loss, reg, inverse_method=None)

    return loss, jac, y, y_hat
