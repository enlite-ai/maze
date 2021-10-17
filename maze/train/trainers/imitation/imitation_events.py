"""Training statistics for imitation learning algorithms."""

from abc import ABC
from typing import Union

import numpy as np

from maze.core.log_stats.event_decorators import define_stats_grouping, define_epoch_stats


class ImitationEvents(ABC):
    """Event interface defining statistics emitted by the imitation learning trainers."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id')
    def policy_loss(self, step_id: Union[str, int], value: float):
        """Optimization loss of the step policy."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id')
    def policy_entropy(self, step_id: Union[str, int], value: float):
        """Entropy of the step policies."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id')
    def policy_l2_norm(self, step_id: Union[str, int], value: float):
        """L2 norm of the step policies."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id')
    def policy_grad_norm(self, step_id: Union[str, int], value: float):
        """Gradient norm of the step policies."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', "subspace_name")
    def discrete_accuracy(self, step_id: Union[str, int], subspace_name: str, value: int):
        """Accuracy for discrete (categorical) subspaces."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', "subspace_name")
    def multi_binary_accuracy(self, step_id: Union[str, int], subspace_name: str, value: int):
        """Accuracy for multi-binary subspaces."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', "subspace_name")
    def box_mean_abs_deviation(self, step_id: Union[str, int], subspace_name: str, value: int):
        """Mean absolute deviation for box (continuous) subspaces."""


class CriticImitationEvents(ABC):
    """Event interface defining statistics emitted by the imitation learning trainers."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id')
    def value(self, step_id: Union[str, int], value: float):
        """Predicted value of the step critics."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id')
    def critic_loss(self, step_id: Union[str, int], value: float):
        """Optimization loss of the step critics."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id')
    def critic_l2_norm(self, step_id: Union[str, int], value: float):
        """L2 norm of the step critics."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id')
    def critic_grad_norm(self, step_id: Union[str, int], value: float):
        """Gradient norm of the step critics."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id')
    def mean_abs_deviation(self, step_id: Union[str, int], value: float):
        """Mean absolute deviation for actual value."""
