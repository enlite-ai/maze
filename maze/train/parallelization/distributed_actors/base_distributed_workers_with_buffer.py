"""Base class for distributing workers with an intermediate buffer to sample transitions from"""

from abc import abstractmethod
from typing import Callable, Optional, Union, Dict, Tuple, List

from omegaconf import DictConfig

from maze.core.agent.policy import Policy
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats import LogStatsAggregator, LogStatsLevel, LogStatsValue, get_stats_logger
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord
from maze.core.utils.factory import Factory
from maze.train.trainers.common.replay_buffer.replay_buffer import BaseReplayBuffer
from maze.train.trainers.common.replay_buffer.uniform_replay_buffer import UniformReplayBuffer


class BaseDistributedWorkersWithBuffer:
    """The base class for all distributed workers with buffer.

    Distributed workers run rollouts independently. Rollouts are collected by calling the collect_rollouts method and
    are then added to the buffer.

    :param env_factory: Factory function for envs to run rollouts on
    :param worker_policy: Structured policy to sample actions from
    :param n_rollout_steps: Number of rollouts steps to record in one rollout
    :param n_workers: Number of distributed workers to run simultaneously
    :param batch_size: Size of the batch the rollouts are collected in
    :param rollouts_per_iteration: The number of rollouts to collect each time the collect_rollouts method is called.
    :param split_rollouts_into_transitions: Specify whether all computed rollouts should be split into
                                            transitions before processing them
    :param env_instance_seeds: A list of seeds for each workers envs.
    :param replay_buffer: The replay buffer to use.
    """

    def __init__(self,
                 env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
                 worker_policy: TorchPolicy,
                 n_rollout_steps: int,
                 n_workers: int,
                 batch_size: int,
                 rollouts_per_iteration: int,
                 split_rollouts_into_transitions: bool,
                 env_instance_seeds: List[int],
                 replay_buffer: BaseReplayBuffer):

        self.env_factory = env_factory
        self._worker_policy = worker_policy
        self.n_rollout_steps = n_rollout_steps
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer

        self.env_instance_seeds = env_instance_seeds

        self.epoch_stats = LogStatsAggregator(LogStatsLevel.EPOCH)
        self.epoch_stats.register_consumer(get_stats_logger('train'))

        self.rollouts_per_iteration = rollouts_per_iteration
        self.split_rollouts_into_transitions = split_rollouts_into_transitions

        self._init_workers()

    @abstractmethod
    def _init_workers(self):
        """Init the agents based on the kind of distribution used"""

    @abstractmethod
    def start(self) -> None:
        """Start all distributed workers"""

    @abstractmethod
    def stop(self) -> None:
        """Stop all distributed workers"""

    def __del__(self) -> None:
        """If the module is deleted stop the workers"""
        self.stop()

    @abstractmethod
    def broadcast_updated_policy(self, state_dict: Dict) -> None:
        """Broadcast the newest version of the policy to the workers.

        :param state_dict: State of the new policy version to broadcast.
        """

    def sample_batch(self, learner_device: str) -> StructuredSpacesRecord:
        """Sample a batch from the buffer and return it as a batched structured spaces record.

        :param learner_device: The device of the learner (cpu or cuda).
        :return: An batched structured spaces record object holding the batched rollouts.
        """
        batch = self.replay_buffer.sample_batch(n_samples=self.batch_size, learner_device=learner_device)
        if self.split_rollouts_into_transitions:
            # Stack records into one, then add an additional dimension
            stacked_records = StructuredSpacesRecord.stack_records(batch)
            return StructuredSpacesRecord.stack_records([stacked_records]).to_torch(learner_device)
        else:
            # Stack trajectories in time major, then stack into a single spaces record
            return SpacesTrajectoryRecord.stack_trajectories(batch).stack().to_torch(learner_device)

    @abstractmethod
    def collect_rollouts(self) -> Tuple[float, float, float]:
        """Collect worker outputs from the queue and add it to the buffer.

        :return: A tuple of (1) queue size before de-queueing,
                 (2) queue size after dequeueing, and (3) the time it took to dequeue the outputs
        """
        raise NotImplementedError

    def get_epoch_stats_aggregator(self) -> LogStatsAggregator:
        """Return the collected epoch stats aggregator"""
        return self.epoch_stats

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
