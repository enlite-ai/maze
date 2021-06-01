"""Implementation of the trivial distribution strategy of calling each environment in sequence in a single thread."""
from typing import List, Callable, Iterable, Any, Tuple, Dict, Optional

import numpy as np

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.log_stats.log_stats import LogStatsLevel
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.train.parallelization.vector_env.structured_vector_env import StructuredVectorEnv
from maze.train.parallelization.vector_env.vector_env import VectorEnv
from maze.train.utils.train_utils import stack_numpy_dict_list, unstack_numpy_list_dict


class SequentialVectorEnv(StructuredVectorEnv):
    """
    Creates a simple wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``, as the overhead of
    multiprocess or multi-thread outweighs the environment computation time. This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_factories: A list of functions that will create the environments
    """

    def __init__(self, env_factories: List[Callable[[], MazeEnv]],
                 logging_prefix: Optional[str] = None):
        self.envs = [LogStatsWrapper.wrap(env_fn()) for env_fn in env_factories]

        super().__init__(
            n_envs=len(env_factories),
            action_spaces_dict=self.envs[0].action_spaces_dict,
            observation_spaces_dict=self.envs[0].observation_spaces_dict,
            agent_counts_dict=self.envs[0].agent_counts_dict,
            logging_prefix=logging_prefix
        )

    def step(self, actions: ActionType) -> Tuple[ObservationType, np.ndarray, np.ndarray, Iterable[Dict[Any, Any]]]:
        """Step the environments with the given actions.

        :param actions: the list of actions for the respective envs.
        :return: observations, rewards, dones, information-dicts all in env-aggregated form.
        """
        actions = unstack_numpy_list_dict(actions)
        observations, rewards, env_dones, infos, actor_dones, actor_ids = [], [], [], [], [], []

        for i, env in enumerate(self.envs):
            o, r, env_done, i = env.step(actions[i])
            actor_dones.append(env.is_actor_done())
            actor_ids.append(env.actor_id())

            if env_done:
                o = env.reset()
                # collect the episode statistics for finished environments
                self.epoch_stats.receive(env.get_stats(LogStatsLevel.EPISODE).last_stats)

            observations.append(o)
            rewards.append(r)
            env_dones.append(env_done)
            infos.append(i)

        obs = stack_numpy_dict_list(observations)
        rewards = np.hstack(rewards).astype(np.float32)
        env_dones = np.hstack(env_dones)

        self._env_times = np.array([env.get_env_time() for env in self.envs])
        self._actor_dones = np.hstack(actor_dones)
        self._actor_ids = actor_ids

        return obs, rewards, env_dones, infos

    @override(StructuredVectorEnv)
    def get_actor_rewards(self) -> Optional[np.ndarray]:
        """Stack actor rewards from encapsulated environments."""
        rewards = [env.get_actor_rewards() for env in self.envs]

        # Return none if rewards are not available
        if rewards[0] is None:
            return None

        rewards = np.stack(rewards, axis=1).astype(np.float32)
        return rewards

    def reset(self) -> Dict[str, np.ndarray]:
        """VectorEnv implementation"""
        observations = []
        for env in self.envs:
            observations.append(env.reset())
            # send the episode statistics of the environment collected before the reset()
            self.epoch_stats.receive(env.get_stats(LogStatsLevel.EPISODE).last_stats)

        self._env_times = np.array([env.get_env_time() for env in self.envs])
        self._actor_ids = [env.actor_id() for env in self.envs]
        self._actor_dones = np.hstack([env.is_actor_done() for env in self.envs])

        return stack_numpy_dict_list(observations)

    @override(VectorEnv)
    def seed(self, seeds: List[Any]) -> None:
        """VectorEnv implementation"""
        assert len(seeds) == len(self.envs)
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def close(self) -> None:
        """VectorEnv implementation"""
        for env in self.envs:
            env.close()
