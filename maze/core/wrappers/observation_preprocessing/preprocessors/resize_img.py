""" Contains an image resizing pre-processor. """
from typing import Tuple, Sequence

import numpy as np
from gym import spaces
from PIL import Image

from maze.core.wrappers.observation_preprocessing.preprocessors.base import PreProcessor
from maze.core.annotations import override


class ResizeImgPreProcessor(PreProcessor):
    """An image resizing pre-processor.

    :param observation_space: The observation space to pre-process.
    :param target_size: Target size of resized image.
    :param transpose: Transpose rgb channel is required (should be last dimension such as [96, 96, 3]).
    """

    def __init__(self, observation_space: spaces.Box, target_size: Sequence[int], transpose: bool):
        super().__init__(observation_space)
        self.target_size = list(target_size)
        self.transpose = transpose

    @override(PreProcessor)
    def processed_shape(self) -> Tuple[int, ...]:
        """implementation of :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        new_shape = list(self._original_observation_space.shape)
        if self.transpose:
            new_shape[-2:] = self.target_size
        else:
            new_shape[:2] = self.target_size
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

        # check if dtype conversion is necessary
        if observation.ndim == 3:
            observation = observation.astype(np.uint8)

        # color channel transposition is necessary
        if self.transpose:
            observation = np.transpose(observation, (1, 2, 0))

        # convert array to image and resize
        img = Image.fromarray(observation)
        img = img.resize(size=self.target_size)

        # convert back to array and transpose back
        observation = np.asarray(img, dtype=np.float32)
        if self.transpose:
            observation = np.transpose(observation, (2, 0, 1))

        return observation
