"""File holding util methods for the simulated environment"""
import logging
from typing import Optional, List, Union, Callable

import numpy as np
from omegaconf import DictConfig

from maze.core.env.maze_env import MazeEnv
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.utils.factory import ConfigType, Factory
from maze.core.utils.seeding import MazeSeeding
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper


def prepare_simulated_env(exclude_wrappers: Optional[List[str]], main_env: MazeEnv, policy_rng: np.random.RandomState,
                          simulated_env: Union[SimulatedEnvMixin, Callable[[], SimulatedEnvMixin], ConfigType]) -> MazeEnv:
    """
    Prepares a simulated environment by excluding certain wrappers, instantiating the env and setting normalization
    statistics.

    :param exclude_wrappers: Wrappers to exclude from simulated environment to reduce overhead.
    :param main_env: The main environment object for copying stats.
    :param policy_rng: A numpy RandomState
    :param simulated_env: A model environment instance used for sampling.
    :return: Instantiated and prepared simulated_env
    """
    # instantiate simulated env from config
    # potentially exclude wrappers from simulated env to be instantiated
    if exclude_wrappers and not isinstance(simulated_env, MazeEnv):
        wrapper_config = DictConfig(simulated_env.wrappers.__dict__["_content"])
        for wrapper in exclude_wrappers:
            if wrapper in wrapper_config:
                logging.info(f"Excluding '{wrapper}' from simulated environment!")
                wrapper_config.pop(wrapper)
        simulated_env.wrappers = wrapper_config

    # instantiate env
    simulated_env = Factory(base_type=MazeEnv).instantiate(simulated_env)
    simulated_env.seed(MazeSeeding.generate_seed_from_random_state(policy_rng))

    # set normalization statistics
    if isinstance(simulated_env, ObservationNormalizationWrapper):
        assert isinstance(main_env, ObservationNormalizationWrapper)
        simulated_env.set_normalization_statistics(main_env.get_statistics())

    return simulated_env
