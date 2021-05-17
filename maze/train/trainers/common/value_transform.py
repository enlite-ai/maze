"""Contains value transformations."""
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import torch
from maze.core.annotations import override


class ValueTransform(ABC):
    """Value transformation (e.g. useful for training the critic in Alpha(Mu)Zero).
    """

    @abstractmethod
    def transform_value(self, x: Union[float, np.ndarray, torch.Tensor]) -> \
            Union[float, np.ndarray, torch.Tensor]:
        """Scale values.

        :param x: Values to be scaled.
        :return: Scaled values.
        """

    @abstractmethod
    def transform_value_inv(self, x: Union[float, np.ndarray, torch.Tensor]) -> \
            Union[float, np.ndarray, torch.Tensor]:
        """Invert scaling of values.

        :param x: Values where scaling should be inverted.
        :return: Values with inverse scaling.
        """


class ReduceScaleValueTransform(ValueTransform):
    """Scale reduction value transform according to Pohlen et al (2018).

    :param epsilon: Lipschitz constant for value function scaling.
    """

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    @override(ValueTransform)
    def transform_value(self, x: Union[float, np.ndarray, torch.Tensor]) -> \
            Union[float, np.ndarray, torch.Tensor]:
        """implementation of :class:`~maze.train.trainers.common.value_transform.ValueTransform` interface
        """
        if isinstance(x, torch.Tensor):
            return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1 + self.epsilon * x)
        else:
            return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1 + self.epsilon * x)

    @override(ValueTransform)
    def transform_value_inv(self, x: Union[float, np.ndarray, torch.Tensor]) -> \
            Union[float, np.ndarray, torch.Tensor]:
        """implementation of :class:`~maze.train.trainers.common.value_transform.ValueTransform` interface
        """
        if isinstance(x, torch.Tensor):
            return torch.sign(x) * (((torch.sqrt(1 + 4 * self.epsilon * (torch.abs(x) + 1 + self.epsilon)) - 1)
                                     / (2 * self.epsilon)) ** 2 - 1)
        else:
            return np.sign(x) * (((np.sqrt(1 + 4 * self.epsilon * (np.abs(x) + 1 + self.epsilon)) - 1)
                                  / (2 * self.epsilon)) ** 2 - 1)


def support_to_scalar(logits: torch.Tensor, support_range: Tuple[int, int]) -> torch.Tensor:
    """Convert support vector to scalar by probability weighted interpolation.

    :param logits: Logits fed into a softmax to get probabilities in range [0, 1].
    :param support_range: Tuple holding the lower and upper bound of the supported value range.
    :return: Tensor of converted scalars.
    """

    # apply softmax
    probabilities = torch.softmax(logits, dim=-1)

    # compile support vector
    support = torch.from_numpy(np.arange(support_range[0], support_range[1] + 1).astype(np.float32))
    support = support.to(device=probabilities.device)

    # compute probability weighted interpolation
    return torch.sum(support * probabilities, dim=-1)


def scalar_to_support(scalar: torch.Tensor, support_range: Tuple[int, int]) -> torch.Tensor:
    """Converts tensor of scalars into probability support vectors corresponding to the provided range.

    :param scalar: Tensor of scalars to be converted
    :param support_range: Tuple holding the lower and upper bound of the supported value range.
    :return: Tensor of support vectors.
    """

    # make sure scalar lives within supported range
    scalar = torch.clamp(scalar, support_range[0], support_range[1])
    floor = scalar.floor()
    prob = scalar - floor

    # initialize tensor of support vectors
    support_set_size = support_range[1] - support_range[0] + 1
    support = torch.zeros(list(scalar.shape) + [support_set_size], device=scalar.device)

    # write lower indices
    lower_indices = (floor - support_range[0]).long()
    indices = torch.clamp(lower_indices, 0, support_set_size - 1)
    support = support.scatter_add(dim=-1, index=indices.unsqueeze(-1), src=1.0 - prob.unsqueeze(-1))

    # write upper indices
    upper_indices = lower_indices + 1
    indices = torch.clamp(upper_indices, 0, support_set_size - 1)
    support = support.scatter_add(dim=-1, index=indices.unsqueeze(-1), src=prob.unsqueeze(-1))

    return support