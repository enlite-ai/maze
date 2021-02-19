"""Contains a dummy observation normalization strategy."""
from typing import List

import numpy as np

from maze.core.annotations import override
from maze.core.wrappers.observation_normalization.normalization_strategies.base import \
    ObservationNormalizationStrategy, StatisticsType


class DummyObservationNormalizationStrategy(ObservationNormalizationStrategy):
    """Dummy normalization strategy.
    """

    @override(ObservationNormalizationStrategy)
    def estimate_stats(self, observations: List[np.ndarray]) -> StatisticsType:
        """Implementation of :class:`~maze.core.wrappers.observation_normalization.observation_normalization_strategy.
        ObservationNormalizationStrategy` interface.
        """

        # compute statistics
        array = np.vstack(observations)
        statistics = {"stat_1": np.max(array, axis=0), "stat_2": np.min(array, axis=0)}

        return statistics

    @override(ObservationNormalizationStrategy)
    def normalize_value(self, value: np.ndarray) -> np.ndarray:
        """Implementation of :class:`~maze.core.wrappers.observation_normalization.observation_normalization_strategy.
        ObservationNormalizationStrategy` interface.
        """
        return value - self._statistics["stat_1"] + self._statistics["stat_2"]
