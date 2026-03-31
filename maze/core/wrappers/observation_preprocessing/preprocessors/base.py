"""Defines interfaces for pre-processor."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces


class PreProcessor(ABC):
    """Interface for observation pre-processors.
    Pre-processors implementing this interface can be used in combination with the
    :class:`~maze.core.wrappers.observation_preprocessing.preprocessing_wrapper.PreProcessingWrapper`.

    :param observation_space: The observation space to pre-process.
    :param kwargs: Arguments to be passed on to preprocessor's constructor.
    """

    def __init__(self, observation_space: spaces.Space, **kwargs):  # noqa: ARG002
        self._original_observation_space = observation_space

    @abstractmethod
    def processed_shape(self) -> tuple[int, ...]:
        """Computes the observation's shape after pre-processing.

        :return: The resulting shape.
        """

    @abstractmethod
    def process(self, observation: np.ndarray) -> np.ndarray:
        """Pre-processes the observation.

        :param observation: The observation to pre-process.
        :return: The pre-processed observation.
        """

    @abstractmethod
    def processed_space(self) -> spaces.Box:
        """Modifies the given observation space according to the respective pre-processor.

        :return: The updated observation space.
        """

    def tag(self) -> str:
        """Returns a tag identifying the pre-processed feature.

        :return: The pre-processor's tag.
        """
        return str(self.__class__).rsplit('.')[-2]
