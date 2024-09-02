from .modeling_layer import (
    Regularizer,
    SquareRegularizer,
    L1Regularizer,
    L2Regularizer,
    HuberRegularizer,
    Sum,
    Loss,
    LogisticLoss,
    MSELoss,
)
from typing import Callable


def regularizer_sum_to_cvxpy(obj, variable):
    match obj:
        case Sum(terms):
            return cp.sum([regularizer_sum_to_cvxpy(term, variable) for term in terms])
        case SquareRegularizer(linear, scale, parameter):
            func = cp.sum_squares
        case L1Regularizer(linear, scale, parameter):
            func = cp.norm1
        case L2Regularizer(linear, scale, parameter):
            func = cp.norm2
        case HuberRegularizer(linear, scale, parameter):
            func = cp.huber
        case _:
            raise RuntimeError("Unknown loss")

    if obj.parameter is None:
        return obj.scale * func(obj.linear @ variable)
    else:
        return obj.scale * obj.parameter * func(obj.linear @ variable)


def loss_to_cvxpy(obj, variable):
    match obj:
        case LogisticLoss(y, X):
            return cp.logistic(-y * X @ variable)
        case MSELoss(y, X):
            return cp.sum_squares(y - X @ variable) / y.numel()
        case _:
            raise RuntimeError("Unknown loss")


def transform_model_to_cvxpy(loss, regularizer, variable):
    return cp.Problem(
        loss_to_cvxpy(loss, variable) + regularizer_sum_to_cvxpy(regularizer, variable)
    )


@dataclass
class Jacobian:
    solution_func: Callable[[], torch.Tensor]
    loss: Loss
    regularizer: Sum | Regularizer

    def __matmul__(self, rhs):
        beta_hat = solution_func
        y = loss.y
        X = loss.X


def transform_model_to_Jacobian(solution_func, loss, regularizer):
    return Jacobian(solution_func, loss, regularizer)
