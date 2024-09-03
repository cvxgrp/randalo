import functools
from typing import Callable, Literal
from dataclasses import dataclass, field

import cvxpy as cp
import linops as lo
import torch

from . import modeling_layer as ml


def regularizer_sum_to_cvxpy(obj, variable):
    match obj:
        case ml.Sum(terms):
            return cp.sum([regularizer_sum_to_cvxpy(term, variable) for term in terms])
        case ml.SquareRegularizer(_):
            func = cp.sum_squares
        case ml.L1Regularizer(_):
            func = cp.norm1
        case ml.L2Regularizer(_):
            func = cp.norm2
        case ml.HuberRegularizer(_):
            func = cp.huber
        case _:
            raise RuntimeError("Unknown loss")

    if obj.parameter is None:
        return obj.scale * func(obj.linear @ variable)
    else:
        return obj.scale * obj.parameter.scale * obj.parameter.parameter * func(obj.linear @ variable)


def loss_to_cvxpy(obj, variable):
    match obj:
        case ml.LogisticLoss(y, X):
            return cp.logistic(-y * X @ variable)
        case ml.MSELoss(y, X):
            return cp.sum_squares(y - X @ variable) / y.numel()
        case _:
            raise RuntimeError("Unknown loss")


def transform_model_to_cvxpy(loss, regularizer, variable):
    return cp.Problem(
        loss_to_cvxpy(loss, variable) + regularizer_sum_to_cvxpy(regularizer, variable)
    )


class Jacobian(lo.LinearOperator):
    solution_func: Callable[[], torch.Tensor]
    loss: ml.Loss
    regularizer: ml.Sum | ml.Regularizer
    inverse_method: Literal[None, 'minres', 'cholesky'] = field(default=None)

    supports_operator_matrix = True

    def __init__(self, solution_func, loss, regularizer, inverse_method):
        self.solution_func = solution_func
        self.loss = loss
        self.regularizer = regularizer
        self.inverse_method = inverse_method
        
    @property
    def _shape(self):
        n = self.loss.y.shape[0]
        return (n, n)


    def _matmul_impl(self, rhs):
        beta_hat = self.solution_func()
        y = utils.to_tensor(self.loss.y)
        X = utils.to_tensor(self.loss.X)
        y, y_hat, dloss_dy_hat, d2loss_dboth, d2loss_dy_hat2 = utils.compute_derivatives(self.loss.func, y, X @ beta_hat)
        

    _diag: torch.Tensor = None

    @functools.cached_property
    def diag(self):
        return torch.diag(self @ torch.eye(self.shape[1]))


def transform_model_to_Jacobian(solution_func, loss, regularizer):
    return Jacobian(solution_func, loss, regularizer)
