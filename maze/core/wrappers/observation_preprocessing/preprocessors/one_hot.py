"""Contains a one hot encoding pre-processor."""

from __future__ import annotations

from maze.core.annotations import override
from maze.core.wrappers.observation_preprocessing.preprocessors.base import PreProcessor

import numpy as np
from gymnasium import spaces


class OneHotPreProcessor(PreProcessor):
    """An one-hot encoding pre-processor for categorical features."""

    @override(PreProcessor)
    def processed_shape(self) -> tuple[int, ...]:
        """implementation of
        :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        if isinstance(self._original_observation_space, spaces.Box):
            high = int(np.max(self._original_observation_space.high) + 1)
            return tuple(list(self._original_observation_space.shape) + [high])
        elif isinstance(self._original_observation_space, spaces.Discrete):
            return (self._original_observation_space.n,)
        else:
            raise ValueError(f'{type(self._original_observation_space)} not supported!')

    @override(PreProcessor)
    def processed_space(self) -> spaces.Box:
        """implementation of
        :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        return spaces.Box(
            low=np.float32(0),
            high=np.float32(1),
            shape=self.processed_shape(),
            dtype=np.float32,
        )

    @override(PreProcessor)
    def process(self, observation: np.ndarray) -> np.ndarray:
        """implementation of
        :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        if isinstance(self._original_observation_space, spaces.Box):
            processed_observation = np.arange(self._original_observation_space.high.max() + 1) == observation[..., None]
            processed_observation = processed_observation.astype(np.float32)
        elif isinstance(self._original_observation_space, spaces.Discrete):
            observation = np.int64(observation)
            processed_observation = np.arange(self._original_observation_space.n) == observation[..., None]
            processed_observation = processed_observation.astype(np.float32)
        else:
            raise ValueError(f'{type(self._original_observation_space)} not supported!')

        assert processed_observation.shape[-len(self.processed_shape()) :] == self.processed_shape()

        return processed_observation
