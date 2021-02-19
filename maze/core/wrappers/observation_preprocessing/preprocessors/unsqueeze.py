""" Contains an un-squeeze pre-processor. """
from typing import Tuple

import numpy as np
from gym import spaces

from maze.core.wrappers.observation_preprocessing.preprocessors.base import PreProcessor
from maze.core.annotations import override


class UnSqueezePreProcessor(PreProcessor):
    """An un-squeeze pre-processor.

    :param observation_space: The observation space to pre-process.
    :param dim: Index where to add an additional dimension.
    """

    def __init__(self, observation_space: spaces.Box, dim: int):
        super().__init__(observation_space)
        self.dim = dim

    @override(PreProcessor)
    def processed_shape(self) -> Tuple[int, ...]:
        """implementation of :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        new_shape = list(self._original_observation_space.shape)
        new_shape.insert(self.dim + 1, 1)
        return tuple(new_shape)

    @override(PreProcessor)
    def processed_space(self) -> spaces.Box:
        """implementation of :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        low = self.process(self._original_observation_space.low)
        high = self.process(self._original_observation_space.high)
        return spaces.Box(low=low, high=high, dtype=self._original_observation_space.dtype)

    @override(PreProcessor)
    def process(self, observation: np.ndarray) -> np.ndarray:
        """implementation of :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        return np.expand_dims(observation, axis=self.dim)
