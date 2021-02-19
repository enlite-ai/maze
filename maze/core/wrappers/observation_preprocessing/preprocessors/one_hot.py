""" Contains a one hot encoding pre-processor. """
from typing import Tuple

import numpy as np
from gym import spaces

from maze.core.wrappers.observation_preprocessing.preprocessors.base import PreProcessor
from maze.core.annotations import override


class OneHotPreProcessor(PreProcessor):
    """An one-hot encoding pre-processor for categorical features.
    """

    @override(PreProcessor)
    def processed_shape(self) -> Tuple[int, ...]:
        """implementation of
        :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        if isinstance(self._original_observation_space, spaces.Box):
            high = int(np.max(self._original_observation_space.high) + 1)
            return tuple(list(self._original_observation_space.shape) + [high])
        elif isinstance(self._original_observation_space, spaces.Discrete):
            return (self._original_observation_space.n,)
        else:
            raise ValueError(f"{type(self._original_observation_space)} not supported!")

    @override(PreProcessor)
    def processed_space(self) -> spaces.Box:
        """implementation of
        :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        return spaces.Box(low=np.float32(0), high=np.float32(1), shape=self.processed_shape(), dtype=np.float32)

    @override(PreProcessor)
    def process(self, observation: np.ndarray) -> np.ndarray:
        """implementation of
        :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        if isinstance(self._original_observation_space, spaces.Box):
            processed_observation = (np.arange(self._original_observation_space.high.max() + 1) == observation[..., None])
            processed_observation = processed_observation.astype(np.float32)
        elif isinstance(self._original_observation_space, spaces.Discrete):
            observation = np.int64(observation)
            processed_observation = (np.arange(self._original_observation_space.n) == observation[..., None])
            processed_observation = processed_observation.astype(np.float32)
        else:
            raise ValueError(f"{type(self._original_observation_space)} not supported!")

        assert processed_observation.shape[-len(self.processed_shape()):] == self.processed_shape()

        return processed_observation
