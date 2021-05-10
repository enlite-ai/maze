"""Contains training utility functions."""
from typing import Union

import numpy as np
import torch


def transform_value(x: Union[float, np.ndarray, torch.Tensor], eps: Union[float, np.ndarray, torch.Tensor]) -> \
        Union[float, np.ndarray, torch.Tensor]:
    """ Scale values in x.

    :param x: Values to be scaled.
    :param eps: Lipschitz constant.
    :return: Scaled values.
    """
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1 + eps * x)
    else:
        return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1 + eps * x)


def transform_value_inv(x: Union[float, np.ndarray, torch.Tensor], eps: Union[float, np.ndarray, torch.Tensor]) -> \
        Union[float, np.ndarray, torch.Tensor]:
    """ Invert scaling of values in x scaled with h().

    :param x: Values where scaling should be inverted.
    :param eps: Lipschitz constant.
    :return: Values with inverse scaling.
    """
    if isinstance(x, torch.Tensor):
        return torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)
    else:
        return np.sign(x) * (((np.sqrt(1 + 4 * eps * (np.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)
