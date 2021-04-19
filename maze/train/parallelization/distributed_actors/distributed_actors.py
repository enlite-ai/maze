"""Base class for distributed actor modules."""

from abc import abstractmethod
from typing import Tuple, Callable, Optional, Union, Dict

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats import LogStatsAggregator, LogStatsLevel, get_stats_logger, LogStatsValue
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord


class DistributedActors:
    """The base class for all distributed actors.

    Distributed actors run rollouts independently. Rollouts are recorded and made available in batches
    to be used during training. When a new policy version is made available, it is distributed to all actors.

    :param env_factory: Factory function for envs to run rollouts on.
    :param policy: Structured policy to sample actions from.
    :param n_rollout_steps: Number of rollouts steps to record in one rollout.
    :param n_actors: Number of distributed actors to run simultaneously.
    :param batch_size: Size of the batch the rollouts are collected in.
    """

    def __init__(self,
                 env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
                 policy: TorchPolicy,
                 n_rollout_steps: int,
                 n_actors: int,
                 batch_size: int):
        self.env_factory = env_factory
        self.policy = policy
        self.n_rollout_steps = n_rollout_steps
        self.n_actors = n_actors
        self.batch_size = batch_size

        self.epoch_stats = LogStatsAggregator(LogStatsLevel.EPOCH)
        self.epoch_stats.register_consumer(get_stats_logger('train'))

    @abstractmethod
    def start(self) -> None:
        """Start all distributed actors"""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop all distributed actors"""
        raise NotImplementedError

    @abstractmethod
    def broadcast_updated_policy(self, state_dict: Dict) -> None:
        """Broadcast the newest version of the policy to the actors.

        :param state_dict: State of the new policy version to broadcast."""
        raise NotImplementedError

    @abstractmethod
    def collect_outputs(self, learner_device: str) -> Tuple[StructuredSpacesRecord, float, float, float]:
        """Collect `self.batch_size` actor outputs from the queue and return them batched where the first dim is
        time and the second is the batch size.

        :param learner_device: the device of the learner
        :return: A tuple of (1) batched version of ActorOutputs, (2) queue size before de-queueing,
                 (3) queue size after dequeueing, and (4) the time it took to dequeue the outputs
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
