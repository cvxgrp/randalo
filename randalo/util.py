from typing import Callable

import numpy as np

import torch
from torch import autograd


def to_tensor(
    array: np.ndarray | torch.Tensor | list[float],
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
    """Convert a numpy array or torch tensor to a torch tensor.

    Parameters
    ----------
    array : np.ndarray | torch.Tensor | list[float]
        Input array or tensor.
    dtype : torch.dtype, optional
        Data type for the output tensor, by default torch.float32.
    device : torch.device, optional
        Device for the output tensor, by default None.

    Returns
    -------
    torch.Tensor
        Output tensor.
    """

    if isinstance(array, np.ndarray):
        return torch.tensor(array, dtype=dtype, device=device)
    elif isinstance(array, torch.Tensor) or isinstance(array, list):
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


def unsqueeze_scalar_like(x: float, array: torch.Tensor) -> torch.Tensor:
    """Expand a scalar to the number of dimensions of an array.

    Parameters
    ----------
    x : float
        Scalar value.
    array : torch.Tensor
        Array to expand the scalar to.

    Returns
    -------
    torch.Tensor
        Expanded scalar.
    """

    return torch.tensor(x, dtype=array.dtype, device=array.device).reshape(
        *(1,) * array.ndim
    )


def create_mixing_matrix(m: int, subsets: list[list[int]]) -> torch.Tensor:
    """Create a mixing matrix A for a given set of subsets, such that when
    M @ A is computed, the columns of M corresponding to the subsets are
    averaged.

    Parameters
    ----------
    m : int
        Number of rows of the mixing matrix.
    subsets : list[list[int]]
        List of subsets of [m] to average.

    Returns
    -------
    torch.Tensor
        Mixing matrix.
    """

    # create mixing matrix
    A = torch.zeros(m, len(subsets))
    for j, subset in enumerate(subsets):
        A[subset, j] = 1 / len(subset)

    return A


def lstsq_y_intercept(
    x: torch.Tensor | np.ndarray | list[float],
    y: torch.Tensor | np.ndarray | list[float],
) -> float:
    """Find the y-intercept of the least squares line.

    Parameters
    ----------
    x : torch.Tensor | np.ndarray | list[float]
        Input data.
    y : torch.Tensor | np.ndarray | list[float]
        Output data.

    Returns
    -------
    float
        The y-intercept of the least squares line.
    """
    x = to_tensor(x).reshape(-1, 1)
    y = to_tensor(y).reshape(-1)
    n = x.shape[0]

    # add intercept
    x = torch.cat([x, torch.ones(n, 1, dtype=x.dtype, device=x.device)], dim=1)

    # solve least squares
    beta, *_ = torch.linalg.lstsq(x, y)

    return beta[-1].item()
