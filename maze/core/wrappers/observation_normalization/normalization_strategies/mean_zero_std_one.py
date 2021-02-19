"""Contains a mean zero standard deviation one observation normalization strategy"""
from typing import List

import numpy as np

from maze.core.annotations import override
from maze.core.wrappers.observation_normalization.normalization_strategies.base import \
    ObservationNormalizationStrategy, StatisticsType


class MeanZeroStdOneObservationNormalizationStrategy(ObservationNormalizationStrategy):
    """Normalizes observations to have zero mean and standard deviation one.

    The strategy first subtracts the observation mean followed by a division with the standard deviation.
    Depending on the original distribution of the input observations this yields a standard Normal.
    """

    @override(ObservationNormalizationStrategy)
    def estimate_stats(self, observations: List[np.ndarray]) -> StatisticsType:
        """Implementation of
        :class:`~maze.core.wrappers.observation_normalization.normalization_strategies.base.ObservationNormalizationStrategy`
        interface.
        """

        # compute statistics
        array = np.vstack(observations)
        keepdims = False if self._axis is None else True
        mean: np.ndarray = np.mean(array, axis=self._axis, keepdims=keepdims)
        std: np.ndarray = np.std(array, axis=self._axis, keepdims=keepdims)

        if keepdims:
            mean = mean[0]
            std = std[0]

        if not isinstance(std, np.ndarray):
            mean = np.asarray(mean, np.float32)
            std = np.asarray(std, np.float32)

        # fix standard deviations to avoid division by zero
        std[std == 0] = 1.0

        statistics = {"mean": mean, "std": std}

        return statistics

    @override(ObservationNormalizationStrategy)
    def normalize_value(self, value: np.ndarray) -> np.ndarray:
        """Implementation of
        :class:`~maze.core.wrappers.observation_normalization.normalization_strategies.base.ObservationNormalizationStrategy`
        interface.
        """

        # check if nan save division is required
        if np.max(np.abs(value)) == np.finfo(np.float32).max:

            # divide by masked broadcasting
            mask = (value != np.finfo(np.float32).min) & (value != np.finfo(np.float32).max)
            mean = np.broadcast_to(self._statistics["mean"], value.shape)
            std = np.broadcast_to(self._statistics["std"], value.shape)

            value[mask] -= mean[mask]
            value[mask] /= std[mask]

        else:
            value = (value - self._statistics["mean"]) / self._statistics["std"]

        return value
