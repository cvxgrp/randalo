import numpy as np
import torch


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
