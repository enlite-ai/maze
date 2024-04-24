"""Training statistics for imitation learning algorithms."""

from abc import ABC
from typing import Union

import numpy as np

from maze.core.log_events.log_create_figure_functions import create_violin_distribution
from maze.core.log_stats.event_decorators import define_stats_grouping, define_epoch_stats, define_plot
from maze.core.log_stats.reducer_functions import histogram


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
    def mean_step_policy_loss(self, value: float):
        """Optimization loss of the step policy."""

    @define_epoch_stats(np.nanmean)
    def mean_step_policy_entropy(self, value: float):
        """Entropy of the step policies."""

    @define_epoch_stats(np.nanmean)
    def mean_step_policy_l2_norm(self, value: float):
        """L2 norm of the step policies."""

    @define_epoch_stats(np.nanmean)
    def mean_step_policy_grad_norm(self, value: float):
        """Gradient norm of the step policies."""

    @define_epoch_stats(np.nanmean)
    def mean_step_discrete_accuracy(self, value: int):
        """Accuracy for discrete (categorical) subspaces."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping("subspace_name")
    def mean_step_discrete_action_rank(self, subspace_name: str, value: int):
        """Rank of target action in discrete (categorical) subspaces."""

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
    def logits(self, substep_key: int, agent_id: int, mean: float, min: float, max: float):
        """Actual network output logits"""

    @define_epoch_stats(np.nanmean, input_name='value', output_name="mean")
    @define_epoch_stats(np.nanmean, input_name='per', output_name="mean_per")
    def logits_greater_zero(self, substep_key: int, agent_id: int, per: float, value: float):
        """When using a support range, this gives the avg number of logits that a greater than 0."""

    @define_epoch_stats(np.nanmean, input_name='value', output_name="mean")
    @define_epoch_stats(np.nanmean, input_name='per', output_name="mean_per")
    def logits_smaller_zero(self, substep_key: int, agent_id: int, per: float, value: float):
        """When using a support range, this gives the avg number of logits that a smaller than 0."""

    @define_epoch_stats(np.nanmean, input_name='mean', output_name="mean")
    @define_epoch_stats(np.nanmin, input_name='min', output_name="min")
    @define_epoch_stats(np.nanmax, input_name='max', output_name="max")
    @define_stats_grouping('substep_key', 'agent_id')
    def actual_value(self, substep_key: int, agent_id: int, mean: float, min: float, max: float):
        """Actual (transformed) value of the step critics."""

    @define_plot(create_figure_function=create_violin_distribution, input_name=None)
    @define_epoch_stats(histogram)
    @define_stats_grouping('substep_key', 'agent_id')
    def abs_diff_hist(self, substep_key: str, agent_id: str, value: float):
        """The absolute distance between the predicted probabilities and the true support for visualization."""

    @define_plot(create_figure_function=create_violin_distribution, input_name=None)
    @define_epoch_stats(histogram)
    @define_stats_grouping('substep_key', 'agent_id')
    def logits_hist(self, substep_key: str, agent_id: str, value: float):
        """A histogram of the actual logits predicted by the network for visualization."""

    @define_plot(create_figure_function=create_violin_distribution, input_name=None)
    @define_epoch_stats(histogram)
    @define_stats_grouping('substep_key', 'agent_id')
    def probs_hist(self, substep_key: str, agent_id: str, value: float):
        """A histogram of the softmax normalized logits predicted by the network for visualization."""

    @define_plot(create_figure_function=create_violin_distribution, input_name=None)
    @define_epoch_stats(histogram)
    @define_stats_grouping('substep_key', 'agent_id')
    def support_hist(self, substep_key: str, agent_id: str, value: float):
        """A histogram of the true target support for visualization."""

    @define_epoch_stats(np.nanmean, input_name='mean', output_name="mean")
    @define_epoch_stats(np.nanmin, input_name='min', output_name="min")
    @define_epoch_stats(np.nanmax, input_name='max', output_name="max")
    @define_stats_grouping('substep_key', 'agent_id')
    def actual_value_original(self, substep_key: int, agent_id: int, mean: float, min: float, max: float):
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
    def value_original(self, substep_key: int, agent_id: int, mean: float, min: float, max: float):
        """Predicted value of the step critics."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('substep_key', 'agent_id')
    def critic_loss(self, substep_key: int, agent_id: int, value: float):
        """Optimization loss of the step critics."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('substep_key', 'agent_id')
    def critic_l2_norm(self, substep_key: int, agent_id: int, value: float):
        """L2 norm of the step critics."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('substep_key', 'agent_id')
    def critic_grad_norm(self, substep_key: int, agent_id: int, value: float):
        """Gradient norm of the step critics."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('substep_key', 'agent_id')
    def mean_abs_deviation(self, substep_key: int, agent_id: int, value: float):
        """Mean absolute deviation for actual value."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('substep_key', 'agent_id')
    def mean_overestimation_deviation(self, substep_key: int, agent_id: int, value: float):
        """Mean overestimation deviation for actual value."""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('substep_key', 'agent_id')
    def mean_underestimation_deviation(self, substep_key: int, agent_id: int, value: float):
        """Mean underestimation deviation for actual value."""

    @define_epoch_stats(np.nanmean, input_name='value', output_name="mean")
    @define_stats_grouping('substep_key', 'agent_id')
    def per_overestimated_wrt_error_greater_01(self, substep_key: int, agent_id: int, value: float):
        """Percent of overestimated values."""

    @define_epoch_stats(np.nanmean, input_name='value', output_name="mean")
    @define_stats_grouping('substep_key', 'agent_id')
    def per_estimation_error_greater_01(self, substep_key: int, agent_id: int, value: float):
        """Percent of sampled value with an error > 1 percent."""

    @define_epoch_stats(np.nanmean)
    def training_iterations(self, value: int):
        """The number of training iterations before early stopping"""
