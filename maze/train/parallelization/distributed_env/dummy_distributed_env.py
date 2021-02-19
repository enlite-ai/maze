"""Implementation of the trivial distribution strategy of calling each environment in sequence in a single thread."""
from typing import List, Callable, Iterable, Any, Tuple, Dict, Union, Optional

import gym
import numpy as np

from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats import LogStatsLevel, LogStatsAggregator, LogStatsValue, get_stats_logger
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.train.parallelization.distributed_env.distributed_env import BaseDistributedEnv
from maze.train.parallelization.observation_aggregator import DictObservationAggregator


class DummyStructuredDistributedEnv(BaseDistributedEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv):
    """
    Creates a simple wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``, as the overhead of
    multiprocess or multi-thread outweighs the environment computation time. This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_factories: A list of functions that will create the environments
        (each callable returns a `MultiStepEnvironment` instance when called).
    """

    def __init__(self, env_factories: List[Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]]],
                 logging_prefix: Optional[str] = None):
        self.envs = [LogStatsWrapper.wrap(env_fn()) for env_fn in env_factories]
        self.num_envs: int = len(self.envs)

        # call super constructor
        BaseDistributedEnv.__init__(self, self.num_envs)

        self.obs_aggregator = DictObservationAggregator()

        self._actor_dones: Optional[np.ndarray] = None
        self._actor_ids: Optional[List[Tuple[Union[str, int], int]]] = None

        # aggregate episode statistics from the workers
        self.epoch_stats = LogStatsAggregator(LogStatsLevel.EPOCH)

        # register a logger for the epoch statistics if desired
        if logging_prefix is not None:
            self.epoch_stats.register_consumer(get_stats_logger(logging_prefix))

    def step(self, actions: List[Any]) -> \
            Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Iterable[Dict[Any, Any]]]:
        """Step the environments with the given actions.

        :param actions: the list of actions for the respective envs.
        :return: observations, rewards, dones, information-dicts all in env-aggregated form.
        """
        self.obs_aggregator.reset()
        rewards, env_dones, infos, actor_dones, actor_ids = [], [], [], [], []

        for i, env in enumerate(self.envs):
            o, r, env_done, i = env.step(actions[i])
            actor_dones.append(env.is_actor_done())
            actor_ids.append(env.actor_id())

            if env_done:
                o = env.reset()
                # collect the episode statistics for finished environments
                self.epoch_stats.receive(env.get_stats(LogStatsLevel.EPISODE).last_stats)

            self.obs_aggregator.observations.append(o)
            rewards.append(r)
            env_dones.append(env_done)
            infos.append(i)

        obs = self.obs_aggregator.aggregate()
        rewards = np.hstack(rewards).astype(np.float32)
        env_dones = np.hstack(env_dones)

        self._actor_dones = np.hstack(actor_dones)
        self._actor_ids = actor_ids

        return obs, rewards, env_dones, infos

    def reset(self) -> Dict[str, np.ndarray]:
        """BaseDistributedEnv implementation"""
        self.obs_aggregator.reset()
        for env in self.envs:
            self.obs_aggregator.observations.append(env.reset())
            # send the episode statistics of the environment collected before the reset()
            self.epoch_stats.receive(env.get_stats(LogStatsLevel.EPISODE).last_stats)

        return self.obs_aggregator.aggregate()

    def seed(self, seed: int = None) -> None:
        """BaseDistributedEnv implementation"""
        for env in self.envs:
            env.seed(seed)
            seed += 1

    def close(self) -> None:
        """BaseDistributedEnv implementation"""
        for env in self.envs:
            env.close()

    @override(LogStatsEnv)
    def get_stats(self, level: LogStatsLevel = LogStatsLevel.EPOCH) -> LogStatsAggregator:
        """Returns the aggregator of the individual episode statistics emitted by the parallel envs.

        :param level: Must be set to `LogStatsLevel.EPOCH`, step or episode statistics are not propagated
        """

        # support only epoch statistics
        assert level == LogStatsLevel.EPOCH

        return self.epoch_stats

    @override(LogStatsEnv)
    def write_epoch_stats(self):
        """Trigger the epoch statistics generation."""
        self.epoch_stats.reduce()

    @override(LogStatsEnv)
    def get_stats_value(self,
                        event: Callable,
                        level: LogStatsLevel,
                        name: Optional[str] = None) -> LogStatsValue:
        """Obtain a single value from the epoch statistics dict.

        :param event: The event interface method of the value in question.
        :param name: The *output_name* of the statistics in case it has been specified in
                     :func:`maze.core.log_stats.event_decorators.define_epoch_stats`
        :param level: Must be set to `LogStatsLevel.EPOCH`, step or episode statistics are not propagated.
        """
        assert level == LogStatsLevel.EPOCH

        return self.epoch_stats.last_stats[(event, name, None)]

    @override(StructuredEnv)
    def actor_id(self) -> List[Tuple[Union[str, int], int]]:
        """Return the actor id tuples of all envs in a list."""
        return self._actor_ids

    @override(StructuredEnv)
    def is_actor_done(self) -> np.ndarray:
        """Return the done flags of all actors in a list."""
        return self._actor_dones

    @property
    @override(StructuredEnvSpacesMixin)
    def action_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """Return the action space of one of the distributed envs."""
        return self.envs[0].action_spaces_dict

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """Return the observation space of one of the distributed envs."""
        return self.envs[0].observation_spaces_dict

    @property
    @override(StructuredEnvSpacesMixin)
    def action_space(self) -> gym.spaces.Dict:
        """implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface
        """
        sub_step_id = self.actor_id[0][0]
        return self.action_spaces_dict[sub_step_id]

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_space(self) -> gym.spaces.Dict:
        """implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface
        """
        sub_step_id = self.actor_id[0][0]
        return self.observation_space[sub_step_id]
