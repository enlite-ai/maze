"""Definition of actor critic training statistics"""
from abc import ABC

import numpy as np

from maze.core.log_stats.event_decorators import define_stats_grouping, define_epoch_stats


class MultiStepActorCriticEvents(ABC):
    """Event interface, defining statistics emitted by the A2CTrainer."""

    @define_epoch_stats(np.mean)
    def time_rollout(self, value: float):
        """time required for rollout"""

    @define_epoch_stats(np.mean)
    def time_epoch(self, value: float):
        """time required for epoch"""

    @define_epoch_stats(np.mean)
    def time_update(self, value: float):
        """time required for update"""

    @define_epoch_stats(np.mean)
    def learning_rate(self, value: float):
        """optimizer learning rate"""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id')
    def policy_loss(self, step_id: int, value: float):
        """optimization loss of the step policy"""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id')
    def policy_grad_norm(self, step_id: int, value: float):
        """gradient norm of the step policies"""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id')
    def policy_entropy(self, step_id: int, value: float):
        """entropy of the step policies"""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('critic_id')
    def critic_value(self, critic_id: int, value: float):
        """critic value of the step critic"""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('critic_id')
    def critic_value_loss(self, critic_id: int, value: float):
        """optimization loss of the step critic"""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('critic_id')
    def critic_grad_norm(self, critic_id: int, value: float):
        """gradient norm of the step critic"""
