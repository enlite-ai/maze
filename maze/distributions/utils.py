"""Contains utils required in the distribution package"""
import torch

EPSILON = 1e-6
MIN_LOG_NN_OUTPUT = -20
MAX_LOG_NN_OUTPUT = 2


def atanh(x: torch.Tensor) -> torch.Tensor:
    """Computes the arc-tangent hyperbolic.

    :param x: The input tensor.
    :return: The arc-tangent hyperbolic of x.
    """
    return 0.5 * torch.log((1 + x) / (1 - x))


def tensor_clamp(x: torch.Tensor, t_min: torch.Tensor, t_max: torch.Tensor) -> torch.Tensor:
    """Clamping with tensor and broadcast support.

    :param x: the tensor to clamp.
    :param t_min: the minimum values.
    :param t_max: the maximum values.
    :return: the clamped tensor.
    """
    return torch.max(torch.min(x, t_max), t_min)
