"""helper functions for ES"""
import torch
from maze.core.agent.torch_policy import TorchPolicy


def get_flat_parameters(policy: TorchPolicy) -> torch.Tensor:
    """Get the parameters of all sub-policies as a single flat vector.

    :param policy: source policy
    :return: flattened parameters
    """
    flat_list = [torch.flatten(p) for p in policy.parameters()]
    flat = torch.cat(flat_list).view(-1)

    return flat


def set_flat_parameters(policy: TorchPolicy, flat_params: torch.Tensor) -> None:
    """Overwrite the parameters of all sub-policies by a single flat vector.

    :param policy: target policy
    :param flat_params: concatenated vector
    """
    start = 0
    for p in policy.parameters():
        end = start + p.numel()
        p[...] = flat_params[start:end].view(*p.shape)
        start = end

