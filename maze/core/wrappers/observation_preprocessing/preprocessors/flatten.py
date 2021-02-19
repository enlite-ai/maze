""" Contains a flattening pre-processor. """
from typing import Tuple

import numpy as np
from gym import spaces

from maze.core.wrappers.observation_preprocessing.preprocessors.base import PreProcessor
from maze.core.annotations import override


class FlattenPreProcessor(PreProcessor):
    """An array flattening pre-processor.

    :param observation_space: The observation space to pre-process.
    :param num_flatten_dims: The number of dimensions to flatten out (from right).
    """

    def __init__(self, observation_space: spaces.Box, num_flatten_dims: int):
        super().__init__(observation_space)
        self.num_flatten_dims = num_flatten_dims

    @override(PreProcessor)
    def processed_shape(self) -> Tuple[int, ...]:
        """implementation of :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        num_flattened_features = np.product(self._original_observation_space.shape[-self.num_flatten_dims:])
        return tuple(list(self._original_observation_space.shape[:-self.num_flatten_dims]) + [num_flattened_features])

    @override(PreProcessor)
    def processed_space(self) -> spaces.Box:
        """implementation of :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        processed_shape = self.processed_shape()

        low = self._original_observation_space.low.copy()
        low = low.reshape(processed_shape)

        high = self._original_observation_space.high.copy()
        high = high.reshape(processed_shape)

        return spaces.Box(low=low, high=high, dtype=self._original_observation_space.dtype)

    @override(PreProcessor)
    def process(self, observation: np.ndarray) -> np.ndarray:
        """implementation of :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        return observation.reshape(self.processed_shape())
