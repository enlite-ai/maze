"""Training statistics for imitation learning algorithms."""

from abc import ABC
from typing import Union

import numpy as np

from maze.core.log_stats.event_decorators import define_stats_grouping, define_epoch_stats


class ImitationEvents(ABC):
    """Event interface defining statistics emitted by the imitation learning trainers."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', 'agent_id')
    def policy_loss(self, step_id: Union[str, int], agent_id: int, value: float):
        """Optimization loss of the step policy."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', 'agent_id')
    def policy_entropy(self, step_id: Union[str, int], agent_id: int, value: float):
        """Entropy of the step policies."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', 'agent_id')
    def policy_l2_norm(self, step_id: Union[str, int], agent_id: int, value: float):
        """L2 norm of the step policies."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', 'agent_id')
    def policy_grad_norm(self, step_id: Union[str, int], agent_id: int, value: float):
        """Gradient norm of the step policies."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', "subspace_name", 'agent_id')
    def discrete_accuracy(self, step_id: Union[str, int], agent_id: int, subspace_name: str, value: int):
        """Accuracy for discrete (categorical) subspaces."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', "subspace_name", 'agent_id')
    def discrete_top_5_accuracy(self, step_id: Union[str, int], agent_id: int, subspace_name: str, value: int):
        """Accuracy for discrete (categorical) subspaces."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', "subspace_name", 'agent_id')
    def discrete_top_10_accuracy(self, step_id: Union[str, int], agent_id: int, subspace_name: str, value: int):
        """Accuracy for discrete (categorical) subspaces."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', "subspace_name", 'agent_id')
    def discrete_action_rank(self, step_id: Union[str, int], agent_id: int, subspace_name: str, value: int):
        """Rank of target action in discrete (categorical) subspaces."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', "subspace_name", 'agent_id')
    def multi_binary_accuracy(self, step_id: Union[str, int], agent_id: int, subspace_name: str, value: int):
        """Accuracy for multi-binary subspaces."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_id', "subspace_name", 'agent_id')
    def box_mean_abs_deviation(self, step_id: Union[str, int], agent_id: int, subspace_name: str, value: int):
        """Mean absolute deviation for box (continuous) subspaces."""

    @define_epoch_stats(np.nanmean)
    def training_iterations(self, value: int):
        """The number of training iterations before early stopping"""


class CriticImitationEvents(ABC):
    """Event interface defining statistics emitted by the imitation learning trainers."""

    @define_epoch_stats(np.nanmean, input_name='mean', output_name="mean")
    @define_epoch_stats(np.nanmin, input_name='min', output_name="min")
    @define_epoch_stats(np.nanmax, input_name='max', output_name="max")
    @define_stats_grouping('substep_key', 'agent_id')
    def actual_value(self, substep_key: int, agent_id: int, mean: float, min: float, max: float):
        """Actual (transformed) value of the step critics."""

    @define_epoch_stats(np.nanmean, input_name='mean', output_name="mean")
    @define_epoch_stats(np.nanmin, input_name='min', output_name="min")
    @define_epoch_stats(np.nanmax, input_name='max', output_name="max")
    @define_stats_grouping('substep_key', 'agent_id')
    def actual_value_original(self, substep_key: int, agent_id: int,  mean: float, min: float, max: float):
        """Actual value of the step critics."""

    @define_epoch_stats(np.nanmean, input_name='mean', output_name="mean")
    @define_epoch_stats(np.nanmin, input_name='min', output_name="min")
    @define_epoch_stats(np.nanmax, input_name='max', output_name="max")
    @define_stats_grouping('substep_key', 'agent_id')
    def value(self, substep_key: int, agent_id: int, mean: float, min: float, max: float):
        """Predicted (transformed) value of the step critics."""

    @define_epoch_stats(np.nanmean, input_name='mean', output_name="mean")
    @define_epoch_stats(np.nanmin, input_name='min', output_name="min")
    @define_epoch_stats(np.nanmax, input_name='max', output_name="max")
    @define_stats_grouping('substep_key', 'agent_id')
    def value_original(self, substep_key: int, agent_id: int,  mean: float, min: float, max: float):
        """Predicted value of the step critics."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('substep_key', 'agent_id')
    def critic_loss(self, substep_key: int, agent_id: int,  value: float):
        """Optimization loss of the step critics."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('substep_key', 'agent_id')
    def critic_l2_norm(self, substep_key: int, agent_id: int,  value: float):
        """L2 norm of the step critics."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('substep_key', 'agent_id')
    def critic_grad_norm(self, substep_key: int, agent_id: int,  value: float):
        """Gradient norm of the step critics."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('substep_key', 'agent_id')
    def mean_abs_deviation(self, substep_key: int, agent_id: int,  value: float):
        """Mean absolute deviation for actual value."""

    @define_epoch_stats(np.nanmean)
    def training_iterations(self, value: int):
        """The number of training iterations before early stopping"""
