"""Adapters for fitted scikit-learn linear models."""

import numbers

import numpy as np
import scipy.sparse
import torch

import sklearn.base
import sklearn.linear_model
import sklearn.utils.class_weight
import sklearn.utils.validation

from . import modeling_layer as ml
from . import reductions


_SUPPORTED_ESTIMATORS = (
    sklearn.linear_model.LinearRegression,
    sklearn.linear_model.Ridge,
    sklearn.linear_model.Lasso,
    sklearn.linear_model.LassoLars,
    sklearn.linear_model.ElasticNet,
    sklearn.linear_model.LogisticRegression,
)


def _as_scalar(value, name):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            f"RandALO only supports scalar {name}; got shape {array.shape}."
        )
    return float(array.reshape(()))


def _validate_sample_weight(sample_weight, n_samples):
    if sample_weight is None:
        return np.ones(n_samples, dtype=float)

    if isinstance(sample_weight, numbers.Real):
        weights = np.full(n_samples, sample_weight, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.ndim == 0:
            weights = np.full(n_samples, weights.item(), dtype=float)
        elif weights.ndim != 1 or weights.shape[0] != n_samples:
            raise ValueError(
                "sample_weight must be a scalar or a one-dimensional array "
                f"with length {n_samples}; got shape {weights.shape}."
            )

    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight must contain only finite values.")
    if np.any(weights < 0):
        raise ValueError("sample_weight cannot contain negative values.")
    if weights.sum() <= 0:
        raise ValueError("sample_weight must sum to a positive value.")
    return weights


def _weighted_loss(loss, weights):
    normalized_weights = weights * (weights.size / weights.sum())
    if np.all(normalized_weights == 1):
        return loss
    return ml.WeightedLoss(loss, normalized_weights)


def _logistic_class_weights(model, y, sample_weight):
    if model.class_weight != "balanced":
        return sklearn.utils.class_weight.compute_sample_weight(model.class_weight, y)

    kwargs = {"class_weight": "balanced", "classes": model.classes_, "y": y}
    try:
        # sklearn >= 1.7 includes sample weights when balancing the classes.
        class_weight = sklearn.utils.class_weight.compute_class_weight(
            **kwargs, sample_weight=sample_weight
        )
    except TypeError as error:
        if "sample_weight" not in str(error):
            raise
        # Older releases computed balanced weights from unweighted counts.
        class_weight = sklearn.utils.class_weight.compute_class_weight(**kwargs)

    weights = np.ones(y.shape[0], dtype=float)
    for label, weight in zip(model.classes_, class_weight):
        weights[y == label] = weight
    return weights


def _coefficient_vector(model, *, logistic=False):
    coef = model.coef_
    if scipy.sparse.issparse(coef):
        coef = coef.toarray()
    coef = np.asarray(coef)

    if logistic:
        if coef.ndim != 2 or coef.shape[0] != 1:
            raise ValueError("Only binary logistic regression is supported.")
        coef = coef[0]
    elif coef.ndim != 1:
        raise ValueError(
            "RandALO only supports single-output regression; "
            f"got coef_ with shape {coef.shape}."
        )
    return coef


def _design_and_solution(model, X, *, logistic=False):
    coef = _coefficient_vector(model, logistic=logistic)
    if coef.shape[0] != X.shape[1]:
        raise ValueError(
            f"X has {X.shape[1]} features but the fitted estimator has "
            f"{coef.shape[0]} coefficients."
        )

    dtype = np.result_type(X.dtype, coef.dtype, float)
    if scipy.sparse.issparse(X):
        X = X.astype(dtype, copy=False)
    else:
        X = np.asarray(X, dtype=dtype)
    penalized = list(range(X.shape[1]))
    if not model.fit_intercept:
        return X, coef.copy(), penalized

    intercept = _as_scalar(model.intercept_, "intercept")
    intercept_scale = 1.0
    penalize_intercept = logistic and model.solver == "liblinear"
    if penalize_intercept:
        # liblinear represents the intercept as a regularized synthetic feature.
        intercept_scale = float(model.intercept_scaling)
        penalized.append(X.shape[1])

    intercept_column = np.full(X.shape[0], intercept_scale, dtype=X.dtype)
    if scipy.sparse.issparse(X):
        X = scipy.sparse.hstack(
            (X, scipy.sparse.csr_matrix(intercept_column[:, None])),
            format="csr",
        )
    else:
        X = np.column_stack((X, intercept_column))
    solution = np.concatenate((coef, [intercept / intercept_scale]))
    return X, solution, penalized


def _zero_or_nonnegative(model, penalized):
    if getattr(model, "positive", False):
        return ml.NonNegativeRegularizer(linear=penalized)
    return ml.ZeroRegularizer()


def _elastic_net_regularizer(l1_scale, square_scale, penalized):
    terms = []
    if l1_scale != 0:
        terms.append(l1_scale * ml.L1Regularizer(linear=penalized))
    if square_scale != 0:
        terms.append(square_scale * ml.SquareRegularizer(linear=penalized))
    if not terms:
        return ml.ZeroRegularizer()
    return sum(terms)


def _logistic_l1_ratio(model):
    """Return the effective L1 ratio across old and new sklearn APIs."""
    penalty = getattr(model, "penalty", "deprecated")
    match penalty:
        case None | "none":
            return None
        case "l1":
            return 1.0
        case "l2":
            return 0.0
        case "elasticnet":
            if model.l1_ratio is None:
                raise ValueError("l1_ratio must be set for an elastic-net penalty.")
            return float(model.l1_ratio)
        case "deprecated":
            # sklearn >= 1.8 expresses the penalty using C and l1_ratio.
            if np.isinf(model.C):
                return None
            return float(model.l1_ratio)
        case _:
            raise ValueError(
                f"Unsupported LogisticRegression penalty: {penalty!r}."
            )


def map_sklearn(
    model: sklearn.base.BaseEstimator = None,
    X: torch.Tensor | np.ndarray = None,
    y: torch.Tensor | np.ndarray | list = None,
    sample_weight: torch.Tensor | np.ndarray | list | float = None,
) -> tuple[ml.Loss, reductions.Jacobian, np.ndarray, np.ndarray]:
    """Map a fitted scikit-learn linear model to RandALO's primitives.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        A fitted supported estimator.
    X : array-like of shape (n_samples, n_features)
        The same dense or CSR/CSC training design matrix used to fit ``model``.
    y : array-like of shape (n_samples,)
        The same training targets used to fit ``model``.
    sample_weight : float or array-like of shape (n_samples,), optional
        The sample weights passed to ``model.fit``. Scikit-learn estimators do
        not retain these weights, so they must be supplied again here.

    Returns
    -------
    tuple
        The loss, Jacobian, encoded targets, and linear predictions.
    """
    if not isinstance(model, _SUPPORTED_ESTIMATORS):
        supported = ", ".join(cls.__name__ for cls in _SUPPORTED_ESTIMATORS)
        raise TypeError(
            f"Unsupported scikit-learn estimator {type(model).__name__}. "
            f"Supported estimators are: {supported}."
        )
    sklearn.utils.validation.check_is_fitted(model, attributes=["coef_"])
    if X is None or y is None:
        raise ValueError("Both X and y must be provided.")

    X_checked = sklearn.utils.validation.check_array(
        X, accept_sparse=("csr", "csc"), ensure_2d=True, dtype="numeric"
    )
    y_checked = np.asarray(y)
    if y_checked.ndim != 1:
        raise ValueError(
            "RandALO only supports one-dimensional targets; "
            f"got y with shape {y_checked.shape}."
        )
    if X_checked.shape[0] != y_checked.shape[0]:
        raise ValueError(
            f"X and y have inconsistent sample counts: {X_checked.shape[0]} "
            f"and {y_checked.shape[0]}."
        )
    n = X_checked.shape[0]
    weights = _validate_sample_weight(sample_weight, n)
    is_logistic = isinstance(model, sklearn.linear_model.LogisticRegression)
    X_design, solution, penalized = _design_and_solution(
        model, X_checked, logistic=is_logistic
    )

    match model:
        case sklearn.linear_model.LinearRegression():
            y_checked = np.asarray(y_checked, dtype=float)
            loss = _weighted_loss(ml.MSELoss(), weights)
            reg = _zero_or_nonnegative(model, penalized)
            y_hat = np.asarray(model.predict(X))

        case sklearn.linear_model.Ridge():
            y_checked = np.asarray(y_checked, dtype=float)
            loss = _weighted_loss(ml.MSELoss(), weights)
            alpha = _as_scalar(model.alpha, "alpha")
            reg = alpha / weights.sum() * ml.SquareRegularizer(linear=penalized)
            if model.positive:
                reg = reg + ml.NonNegativeRegularizer(linear=penalized)
            y_hat = np.asarray(model.predict(X))

        case sklearn.linear_model.LassoLars():
            if sample_weight is not None:
                raise ValueError("LassoLars.fit does not support sample_weight.")
            y_checked = np.asarray(y_checked, dtype=float)
            loss = ml.MSELoss()
            alpha = _as_scalar(model.alpha, "alpha")
            reg = _elastic_net_regularizer(2.0 * alpha, 0.0, penalized)
            if model.positive and alpha == 0:
                reg = ml.NonNegativeRegularizer(linear=penalized)
            y_hat = np.asarray(model.predict(X))

        case sklearn.linear_model.Lasso():
            y_checked = np.asarray(y_checked, dtype=float)
            loss = _weighted_loss(ml.MSELoss(), weights)
            alpha = _as_scalar(model.alpha, "alpha")
            reg = _elastic_net_regularizer(2.0 * alpha, 0.0, penalized)
            if model.positive and alpha == 0:
                reg = ml.NonNegativeRegularizer(linear=penalized)
            y_hat = np.asarray(model.predict(X))

        case sklearn.linear_model.ElasticNet():
            y_checked = np.asarray(y_checked, dtype=float)
            loss = _weighted_loss(ml.MSELoss(), weights)
            alpha = _as_scalar(model.alpha, "alpha")
            reg = _elastic_net_regularizer(
                2.0 * alpha * model.l1_ratio,
                alpha * (1.0 - model.l1_ratio),
                penalized,
            )
            if model.positive and alpha == 0:
                reg = ml.NonNegativeRegularizer(linear=penalized)
            y_hat = np.asarray(model.predict(X))

        case sklearn.linear_model.LogisticRegression():
            if len(model.classes_) != 2:
                raise ValueError("Only binary logistic regression is supported.")
            negative = y_checked == model.classes_[0]
            positive = y_checked == model.classes_[1]
            if not np.all(negative | positive):
                raise ValueError("y contains labels not present in model.classes_.")

            class_weights = _logistic_class_weights(model, y_checked, weights)
            weights = weights * class_weights
            if weights.sum() <= 0:
                raise ValueError(
                    "The combined sample and class weights must be positive."
                )
            loss = _weighted_loss(ml.LogisticLoss(), weights)
            y_checked = np.where(positive, 1.0, -1.0)

            l1_ratio = _logistic_l1_ratio(model)
            inverse_strength = 0.0 if np.isinf(model.C) else 1.0 / model.C
            if l1_ratio is None or inverse_strength == 0:
                reg = ml.ZeroRegularizer()
            else:
                reg = _elastic_net_regularizer(
                    inverse_strength * l1_ratio / weights.sum(),
                    0.5 * inverse_strength * (1.0 - l1_ratio) / weights.sum(),
                    penalized,
                )
            y_hat = np.asarray(model.decision_function(X))

        case _:
            raise AssertionError("Validated estimator was not mapped.")

    if y_hat.ndim != 1:
        raise ValueError(
            "RandALO only supports estimators with one-dimensional predictions; "
            f"got predictions with shape {y_hat.shape}."
        )

    jac = reductions.Jacobian(
        y_checked,
        X_design,
        lambda: solution,
        loss,
        reg,
        inverse_method=None,
    )
    return loss, jac, y_checked, y_hat
