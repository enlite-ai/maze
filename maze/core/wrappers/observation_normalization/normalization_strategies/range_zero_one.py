"""Contains a  range [0, 1] observation normalization strategy"""
from typing import List

import numpy as np

from maze.core.annotations import override
from maze.core.wrappers.observation_normalization.normalization_strategies.base import \
    ObservationNormalizationStrategy, StatisticsType


class RangeZeroOneObservationNormalizationStrategy(ObservationNormalizationStrategy):
    """Normalizes observations to value range [0, 1].

    The strategy subtracts in a first step the minimum observed value to shift the lowest value after
    normalization to zero. In a subsequent step we divide the observation with the maximum of the previous step yielding
    observations in the range [0, 1].
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
        min_val: np.ndarray = np.min(array, axis=self._axis, keepdims=keepdims)
        max_val: np.ndarray = np.max(array, axis=self._axis, keepdims=keepdims)

        if keepdims:
            min_val = min_val[0]
            max_val = max_val[0]

        if not isinstance(min_val, np.ndarray):
            min_val = np.asarray(min_val, np.float32)
            max_val = np.asarray(max_val, np.float32)

        statistics = {"min": min_val, "max": max_val}

        return statistics

    @override(ObservationNormalizationStrategy)
    def normalize_value(self, value: np.ndarray) -> np.ndarray:
        """Implementation of
        :class:`~maze.core.wrappers.observation_normalization.normalization_strategies.base.ObservationNormalizationStrategy`
        interface.
        """
        return (value - self._statistics["min"]) / (self._statistics["max"] - self._statistics["min"])
