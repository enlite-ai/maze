"""This file contains the method for creating the rllib compatible maze env factory."""
from typing import Callable

from omegaconf import DictConfig

from maze.core.utils.config_utils import EnvFactory
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.no_dict_action_wrapper import NoDictActionWrapper
from maze.core.wrappers.observation_normalization.observation_normalization_utils import \
    obtain_normalization_statistics, make_normalized_env_factory
from maze.core.wrappers.wrapper import EnvType


def build_maze_rllib_env_factory(cfg: DictConfig) -> Callable[[], EnvType]:
    """Create the rllib compatible maze env-factory, after applying rllib monkey patches

    :param cfg: The hydra config.
    """

    def env_factory() -> EnvType:
        """Create a new env in order to apply the monkey patch in worker-process.
        :return: A env factory
        """
        env = EnvFactory(cfg.env, cfg.wrappers if 'wrappers' in cfg else {})()
        logging_env = LogStatsWrapper.wrap(env)

        return logging_env

    # Observation normalization
    normalization_env = env_factory()
    normalization_statistics = obtain_normalization_statistics(normalization_env,
                                                               n_samples=cfg.runner.normalization_samples)
    if normalization_statistics:
        env_factory = make_normalized_env_factory(env_factory, normalization_statistics)

    def final_env_factory() -> EnvType:
        """Create the env with the NoDictWrapper applied if needed.
        :return: A env factory
        """
        env = env_factory()
        if cfg.algorithm.model_cls == 'maze.rllib.maze_rllib_models.maze_rllib_q_model.MazeRLlibQModel':
            env = NoDictActionWrapper(env)
        return env

    return final_env_factory
