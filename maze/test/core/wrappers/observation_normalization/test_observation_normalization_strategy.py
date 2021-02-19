"""Contains observation normalization strategy tests"""

import gym
import numpy as np

from maze.core.wrappers.observation_normalization.normalization_strategies.mean_zero_std_one import \
    MeanZeroStdOneObservationNormalizationStrategy
from maze.core.wrappers.observation_normalization.normalization_strategies.range_zero_one import \
    RangeZeroOneObservationNormalizationStrategy


def get_structured_space():
    """ helper function """

    space_dict = dict()
    space_dict["box"] = gym.spaces.Box(low=np.full(shape=(10,), fill_value=0, dtype=np.float32),
                                       high=np.full(shape=(10,), fill_value=100, dtype=np.float32),
                                       dtype=np.float32)

    observation_space_0 = gym.spaces.Dict(spaces=space_dict)

    return {0: observation_space_0}


def test_normalization_strategies():
    """ normalization strategy test """

    for normalization_strategy_cls in [MeanZeroStdOneObservationNormalizationStrategy,
                                       RangeZeroOneObservationNormalizationStrategy]:

        structured_space = get_structured_space()

        observation_space = structured_space[0]["box"]
        normalization_strategy = normalization_strategy_cls(observation_space, clip_range=(0, 1), axis=0)
        assert isinstance(normalization_strategy, normalization_strategy_cls)
        assert normalization_strategy.is_initialized() is False

        observations = [observation_space.sample() for _ in range(100)]
        stats = normalization_strategy.estimate_stats(observations)
        normalization_strategy.set_statistics(stats)
        assert normalization_strategy.is_initialized() is True
        for key, value in normalization_strategy.get_statistics().items():
            assert np.all(stats[key] == value)

        normalized_observation_space = normalization_strategy.normalized_space()
        assert np.all(normalized_observation_space.low >= 0)
        assert np.all(normalized_observation_space.high <= 1)

        normalized_observation = normalization_strategy.normalize_and_process_value(observations[0])
        assert normalized_observation in normalized_observation
