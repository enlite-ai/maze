""" Contains utility functions to be used in concert with the ObservationNormalizationWrapper """

import os
from typing import Union, Optional, Callable

from tqdm import tqdm

from maze.core.env.maze_env import MazeEnv
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.wrappers.observation_normalization.normalization_strategies.base import StructuredStatisticsType
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper


def estimate_observation_normalization_statistics(env: Union[MazeEnv, ObservationNormalizationWrapper],
                                                  n_samples: int) -> None:
    """Helper function estimating normalization statistics.
    :param env: The observation normalization wrapped environment.
    :param n_samples: The number of samples (i.e., flat environment steps) to take for statistics computation.
    """

    # remove previous statistics dump
    if os.path.exists(env.statistics_dump):
        os.remove(env.statistics_dump)

    print(f'******* Starting to estimate observation normalization statistics for {n_samples} steps *******')
    # collect normalization statistics
    env.set_observation_collection(True)
    rollout_generator = RolloutGenerator(env)

    for _ in tqdm(range(n_samples)):
        rollout_generator.rollout(policy=env.sampling_policy, n_steps=1)

    # finally estimate normalization statistics
    env.estimate_statistics()


def obtain_normalization_statistics(env: Union[MazeEnv, ObservationNormalizationWrapper], n_samples: int) \
        -> Optional[StructuredStatisticsType]:
    """Obtain the normalization statistics of a given environment.

    * Returns None, if the ObservationNormalizationWrapper is not implemented
    * Returns the loaded statistics, if available
    * Runs the estimation and returns the newly calculated statistics, if not loaded previously

    :param env: Environment with applied ObservationNormalizationWrapper (function returns None immediately if this is
        not the case.
    :param n_samples: Number of samples (=steps) to collect normalization statistics at the beginning of
        the training.
    :return: The normalization statistics or None if the ObservationNormalizationWrapper is not implemented by the env.
    """
    if not isinstance(env, ObservationNormalizationWrapper):
        return None

    # first check if loading from file was successful
    normalization_statistics = env.loaded_stats
    if normalization_statistics:
        return normalization_statistics

    # no statistics available, run estimation
    estimate_observation_normalization_statistics(env, n_samples=n_samples)
    normalization_statistics = env.get_statistics()

    # deactivate statistics collection
    env.set_observation_collection(False)

    return normalization_statistics


def make_normalized_env_factory(
        env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, ObservationNormalizationWrapper]],
        normalization_statistics: StructuredStatisticsType
) -> Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, ObservationNormalizationWrapper]]:
    """Wrap an existing env factory to assign the passed normalization statistics.

    :param env_factory: The existing env factory
    :param normalization_statistics: The normalization statistics that should be applied to the env
    :return: The wrapped env factory
    """

    def normalized_env_factory() -> Union[StructuredEnv, StructuredEnvSpacesMixin, ObservationNormalizationWrapper]:
        """the wrapped env factory"""
        env = env_factory()
        env.set_normalization_statistics(normalization_statistics)

        return env

    return normalized_env_factory
