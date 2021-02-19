"""Definition of Evolution Strategies training statistics"""
import numpy as np

from maze.core.log_stats.event_decorators import define_epoch_stats, define_stats_grouping


class ESEvents:
    """Event interface, defining statistics emitted by the ESTrainer."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('policy_id')
    def policy_grad_norm(self, policy_id: int, value: float):
        """gradient norm of the step policies"""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('policy_id')
    def policy_norm(self, policy_id: int, value: float):
        """l2 norm of the step policy parameters"""

    @define_epoch_stats(np.sum, output_name="real_time")
    @define_epoch_stats(np.sum, output_name="total_real_time", cumulative=True)
    def real_time(self, value: float):
        """elapsed real time per iteration (=epoch)"""

    @define_epoch_stats(np.mean)
    def update_ratio(self, value: float):
        """norm(optimizer step) / norm(all parameters)"""
