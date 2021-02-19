""" Contains a transpose pre-processor. """
from typing import Tuple, Sequence

import numpy as np
from gym import spaces

from maze.core.wrappers.observation_preprocessing.preprocessors.base import PreProcessor
from maze.core.annotations import override


class TransposePreProcessor(PreProcessor):
    """An array transposition pre-processor.

    :param observation_space: The observation space to pre-process.
    :param axes: The num ordering of the axes of the input array.
    """

    def __init__(self, observation_space: spaces.Box, axes: Sequence[int]):
        super().__init__(observation_space)
        self.axes = list(axes)

    @override(PreProcessor)
    def processed_shape(self) -> Tuple[int, ...]:
        """implementation of :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        new_shape = [self._original_observation_space.shape[i] for i in self.axes]
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
        return np.transpose(observation, axes=self.axes)
