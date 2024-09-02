from typing import Callable

import numpy as np

import torch
from torch import autograd


def to_tensor(
    array: np.ndarray | torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:

    if isinstance(array, np.ndarray):
        return torch.tensor(array, dtype=dtype, device=device)
    elif isinstance(array, torch.Tensor):
        return torch.as_tensor(array, dtype=dtype, device=device)
    else:
        raise ValueError("Input must be a numpy array or torch tensor")


def compute_derivatives(
    loss_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    y_hat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute first and second derivatives of a loss function.

    Parameters
    ----------
    loss_fun : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function to compute derivatives of. The function should take the
        true labels `y` and the predicted values `y_hat` as input and return
        the element-wise loss.

    y : torch.Tensor
        True labels.

    y_hat : torch.Tensor
        Predicted values.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        The true labels,
        the predicted values,
        the first derivatives of the loss with respect to the predicted values,
        the second derivatives of the loss with respect to both the true labels
            and predicted values,
        and the second derivatives of the loss with respect to the predicted values.
    """

    # detach and clone to avoid memory leaks
    y = y.detach().clone().requires_grad_(True)
    y_hat = y_hat.detach().clone().requires_grad_(True)
    n = y.shape[0]

    # compute first and second derivatives of loss function
    # we obtain the vector derivatives by summing and then taking the gradient
    loss = loss_fun(y, y_hat).sum()
    # keep the graph for computing the second derivatives
    dloss_dy_hat, *_ = autograd.grad(loss, y_hat, create_graph=True)
    dloss_dy_hat_sum = dloss_dy_hat.sum()

    d2loss_dboth, d2loss_dy_hat2 = autograd.grad(
        dloss_dy_hat_sum, [y, y_hat], allow_unused=True
    )
    if d2loss_dboth is None:
        d2loss_dboth = torch.zeros_like(y)
    if d2loss_dy_hat2 is None:
        d2loss_dy_hat2 = torch.zeros_like(y_hat)

    # free memory used by autograd and return
    return (
        y.detach(),
        y_hat.detach(),
        dloss_dy_hat.detach(),
        d2loss_dboth.detach(),
        d2loss_dy_hat2.detach(),
    )
