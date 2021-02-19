"""Contains interface definitions for observation normalization strategies."""
import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Iterable, Tuple, Optional

import gym
import numpy as np

# specify data types
Number = Union[float, int]
StatisticsType = Dict[str, Union[np.ndarray, Number, Iterable[Number]]]
StructuredStatisticsType = Dict[str, StatisticsType]


class ObservationNormalizationStrategy(ABC):
    """Abstract base class for normalization strategies.

    Provides functionality for:
        - normalizing gym.Box observations as well as for normalizing the originally defined observation space.
        - setting and getting the currently employed normalization statistics.
        - interface definition for estimating the normalization statistics from a list of observations
        - interface definition for normalizing a given gym.Box (np.ndarray) observation

    :param observation_space: The observations space to be normalized.
    :param clip_range: The minimum and maximum value allowed for an observation.
    :param axis: Defines the axis along which to compute normalization statistics
    """

    def __init__(self, observation_space: gym.spaces.Box, clip_range: Tuple[Number, Number],
                 axis: Optional[Union[int, Tuple[int], List[int]]]):

        assert isinstance(observation_space, gym.spaces.Box)
        self._original_observation_space = copy.deepcopy(observation_space)
        self._statistics: Optional[StatisticsType] = None
        self._axis = axis
        # Convert to tuple, since yaml only reads lists
        if isinstance(self._axis, Iterable):
            self._axis = tuple(self._axis)

        # set allowed value ranges
        self._clip_min = clip_range[0]
        self._clip_max = clip_range[1]

    def normalized_space(self) -> gym.spaces.Box:
        """Normalizes extrema (low and high) in the observation space with respect to the given statistics.
        (e.g. it sets the maximum value of a Box space to the maximum in the respective observation)

        :return: Observation space with extrema adjusted w.r.t. statistics and normalization strategy.
        """
        space = copy.deepcopy(self._original_observation_space)
        space.low = self.normalize_and_process_value(space.low)
        space.high = self.normalize_and_process_value(space.high)
        return space

    def normalize_and_process_value(self, value: np.ndarray) -> np.ndarray:
        """Normalizes and post-processes the actual observation (see also: normalize_value).

        :param value: Observation value to be normalized.
        :return: Normalized and processed observation.
        """

        # apply actual value normalization
        value = self.normalize_value(value)

        # clip value to pre-defined range
        if self._clip_min is not None or self._clip_max is not None:
            value = np.clip(value, a_min=self._clip_min, a_max=self._clip_max)

        return value

    def set_statistics(self, stats: StatisticsType) -> None:
        """Set normalization statistics.

        :param stats: A dictionary containing the respective observation normalization statistics.
        """

        results = dict()

        for stats_key, values in stats.items():
            results[stats_key] = np.asarray(values, dtype=np.float32)

            # assert that broadcasting is possible with provided statistics
            obs_shape = self._original_observation_space.shape
            assert np.broadcast(np.ones(shape=obs_shape), results[stats_key]).shape == obs_shape

        self._statistics = results

    def get_statistics(self) -> StatisticsType:
        """Get normalization statistics.

        :return: The normalization statistics.
        """
        return self._statistics

    def is_initialized(self) -> bool:
        """Checks if the normalization strategy is fully initialized.

        :return: True if fully initialized and ready to normalize; else False.
        """
        return self._statistics is not None

    @abstractmethod
    def estimate_stats(self, observations: List[np.ndarray]) -> StatisticsType:
        """Estimate observation statistics from collected observations.

        :param observations: A lists of observations.
        """

    @abstractmethod
    def normalize_value(self, value: np.ndarray) -> np.ndarray:
        """Normalizes the actual observation value with provided statistics.
        The type and shape of value and statistics have to match.

        :param value: Observation to be normalized.
        :return: Normalized observation.
        """
