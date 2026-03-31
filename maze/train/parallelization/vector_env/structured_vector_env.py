from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from maze.core.annotations import override
from maze.core.env.structured_env import ActorID, StepKeyType, StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.env.time_env_mixin import TimeEnvMixin
from maze.core.log_stats.log_stats import LogStatsAggregator, LogStatsLevel, LogStatsValue, get_stats_logger
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.train.parallelization.vector_env.vector_env import VectorEnv

import gymnasium as gym
import numpy as np


class StructuredVectorEnv(VectorEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv, TimeEnvMixin, ABC):
    """Common superclass for the structured vectorised env implementations in Maze.

    :param n_envs: The number of vectorised environments.
    :param action_spaces_dict: Action spaces dict (not vectorized, as it is the same for all environments)
    :param observation_spaces_dict: Observation spaces dict (not vectorized, as it is the same for all environments)
    :param logging_prefix: If set, will report epoch statistics under this logging prefix.
    """

    def __init__(
        self,
        n_envs: int,
        action_spaces_dict: dict[StepKeyType, gym.spaces.Space],
        observation_spaces_dict: dict[StepKeyType, gym.spaces.Space],
        agent_counts_dict: dict[StepKeyType, int],
        logging_prefix: str | None = None,
    ):
        super().__init__(n_envs)

        # Spaces
        self._action_spaces_dict = action_spaces_dict
        self._observation_spaces_dict = observation_spaces_dict
        self._agent_counts_dict = agent_counts_dict

        # Aggregate episode statistics from individual envs
        self.epoch_stats = LogStatsAggregator(LogStatsLevel.EPOCH)

        # register a logger for the epoch statistics if desired
        if logging_prefix is not None:
            self.epoch_stats.register_consumer(get_stats_logger(logging_prefix))

        # Keep track of current actor IDs, actor dones, and env times (should be updated in step and reset methods).
        self._actor_ids = None
        self._actor_dones = None
        self._env_times = None

        # to help determine done in Gym 26
        self._actor_terminated = None
        self._actor_truncated = None

        self.seeds = None
        self._next_seed_idx = 0

    def get_next_seed(self) -> Any | None:
        """Return the next seed to use.

        :return: The next seed to use.
        """
        if self.seeds is None:
            return None

        seed = self.seeds[self._next_seed_idx]
        self._next_seed_idx = (self._next_seed_idx + 1) % len(self.seeds)
        return seed

    @override(StructuredEnv)
    def actor_id(self) -> ActorID:
        """Current actor ID (should be the same for all envs, as only synchronous envs are supported)."""
        assert len(set(self._actor_ids)) == 1, 'only synchronous environments are supported.'
        return self._actor_ids[0]

    @property
    def agent_counts_dict(self) -> dict[StepKeyType, int]:
        """Return the agent counts of one of the vectorised envs."""
        return self._agent_counts_dict

    @override(StructuredEnv)
    def is_actor_done(self) -> np.ndarray:
        """Return the done flags (terminated or truncated) of all actors in a list."""
        return self._actor_terminated | self._actor_truncated

    def is_actor_terminated(self) -> bool:
        """Return the terminated flags of all actors in a list."""
        return self._actor_terminated

    def is_actor_truncated(self) -> bool:
        """Return the truncated flags of all actors in a list."""
        return self._actor_truncated

    @abstractmethod
    @override(StructuredEnv)
    def get_actor_rewards(self) -> np.ndarray | None:
        """Individual implementations need to override this to support structured rewards."""

    @override(TimeEnvMixin)
    def get_env_time(self) -> np.ndarray:
        """Return current env time for all vectorised environments."""
        return self._env_times

    @property
    def action_spaces_dict(self) -> dict[int | str, gym.spaces.Space]:
        """Return the action space of one of the vectorised envs."""
        return self._action_spaces_dict

    @property
    def observation_spaces_dict(self) -> dict[int | str, gym.spaces.Space]:
        """Return the observation space of one of the vectorised envs."""
        return self._observation_spaces_dict

    @property
    @override(StructuredEnvSpacesMixin)
    def action_space(self) -> gym.spaces.Space:
        """implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface"""
        sub_step_id, _ = self.actor_id()
        return self.action_spaces_dict[sub_step_id]

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_space(self) -> gym.spaces.Space:
        """implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface"""
        sub_step_id, _ = self.actor_id()
        return self.observation_spaces_dict[sub_step_id]

    @override(LogStatsEnv)
    def get_stats(self, level: LogStatsLevel) -> LogStatsAggregator:
        """Returns the aggregator of the individual episode statistics emitted by the parallel envs.

        :param level: Must be set to `LogStatsLevel.EPOCH`, step or episode statistics are not propagated
        """
        assert level == LogStatsLevel.EPOCH
        return self.epoch_stats

    @override(LogStatsEnv)
    def write_epoch_stats(self):
        """Trigger the epoch statistics generation."""
        self.epoch_stats.reduce()

    @override(LogStatsEnv)
    def clear_epoch_stats(self) -> None:
        """Clear out episode statistics collected so far in this epoch."""
        self.epoch_stats.clear_inputs()

    @override(LogStatsEnv)
    def get_stats_value(self, event: Callable, level: LogStatsLevel, name: str | None = None) -> LogStatsValue:
        """Obtain a single value from the epoch statistics dict.

        :param event: The event interface method of the value in question.
        :param name: The *output_name* of the statistics in case it has been specified in
                     :func:`maze.core.log_stats.event_decorators.define_epoch_stats`
        :param level: Must be set to `LogStatsLevel.EPOCH`, step or episode statistics are not propagated.
        """
        assert level == LogStatsLevel.EPOCH
        return self.epoch_stats.last_stats[(event, name, None)]
