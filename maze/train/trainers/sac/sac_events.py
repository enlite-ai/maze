"""SAC Events"""

from abc import ABC
from typing import Union

import numpy as np

from maze.core.log_stats.event_decorators import define_stats_grouping, define_epoch_stats


class SACEvents(ABC):
    """Events specific for the SAC algorithm, in order to record and analyse it's behaviour in more detail"""

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_key')
    def policy_loss(self, step_key: Union[int, str], value: float) -> None:
        """Record the policy loss.

        :param step_key: The step_key of the multi-step env.
        :param value: The value.
        """

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_key')
    def policy_grad_norm(self, step_key: Union[int, str], value: float) -> None:
        """Record the gradient norm.

        :param step_key: The step_key of the multi-step env.
        :param value: The value.
        """

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_key')
    def policy_entropy(self, step_key: Union[int, str], value: float) -> None:
        """Record the policy entropy.

        :param step_key: The step_key of the multi-step env.
        :param value: The value.
        """

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_key')
    def policy_mean_logp(self, step_key: Union[int, str], value: float) -> None:
        """Record the mean policy logp.

        :param step_key: The step_key of the multi-step env.
        :param value: The value.
        """

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('critic_key')
    def errors_between_critics(self, critic_key: Union[int, str], value: float) -> None:
        """Record the error between critic and target critic.

        :param critic_key: The key of the critic.
        :param value: The value.
        """

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('critic_key')
    def critic_value(self, critic_key: Union[int, str], value: float) -> None:
        """Record the critic value.

        :param critic_key: The key of the critic.
        :param value: The value.
        """

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('critic_key')
    def critic_value_loss(self, critic_key: Union[int, str], value: float) -> None:
        """Record the critic value loss.

        :param critic_key: The key of the critic.
        :param value: The value.
        """

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('critic_key')
    def critic_grad_norm(self, critic_key: Union[int, str], value: float) -> None:
        """Record the critic gradient norm.

        :param critic_key: The key of the critic.
        :param value: The value.
       """

    @define_epoch_stats(np.nanmean, input_name='time')
    @define_epoch_stats(np.nanmean, input_name='percent')
    def time_dequeuing_actors(self, time: float, percent: float) -> None:
        """Record the time it took to dequeue the actors output from the synced queue + relative per to total update
        time.

        :param time: The absolute time it took for the computation.
        :param percent: The relative percentage this computation took w.r.t. to one update.
        """

    @define_epoch_stats(np.nanmean, input_name='time')
    @define_epoch_stats(np.nanmean, input_name='percent')
    def time_learner_rollout(self, time: float, percent: float) -> None:
        """Record the total time it took the learner to compute the logits on the agents output
            + relative per to total update time.

        :param time: The absolute time it took for the computation.
        :param percent: The relative percentage this computation took w.r.t. to one update.
        """

    @define_epoch_stats(np.nanmean, input_name='time')
    @define_epoch_stats(np.nanmean, input_name='percent')
    def time_loss_computation(self, time: float, percent: float) -> None:
        """Record the total time it took the learner compute the loss + relative per to total update time.

        :param time: The absolute time it took for the computation.
        :param percent: The relative percentage this computation took w.r.t. to one update.
        """

    @define_epoch_stats(np.nanmean, input_name='time')
    @define_epoch_stats(np.nanmean, input_name='percent')
    def time_backprob(self, time: float, percent: float) -> None:
        """Record the total time it took the learner to backprob the loss + relative per to total update time.

        :param time: The absolute time it took for the computation.
        :param percent: The relative percentage this computation took w.r.t. to one update.
        """

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_key')
    def entropy_coef(self, step_key: Union[str, int], value: float) -> None:
        """Record the current entropy coefficient, interesting when using entropy tuning.

        :param step_key: The step_key of the multi-step env.
        :param value: The current value of the entropy coefficient.
        """

    @define_epoch_stats(np.nanmean)
    @define_stats_grouping('step_key')
    def entropy_loss(self, step_key: Union[str, int], value: float) -> None:
        """Record the current entropy loss, interesting when using entropy tuning.

        :param step_key: The step_key of the multi-step env.
        :param value: The current value of the entropy loss.
        """

    @define_epoch_stats(np.nanmean, input_name='before')
    @define_epoch_stats(np.nanmean, input_name='after')
    def estimated_queue_sizes(self, before: int, after: int) -> None:
        """Record the estimated queue size before and after the collection of the actors output.

        :param before: The estimated queue size before collection.
        :param after: The estimated queue size after collection.
        """

    @define_epoch_stats(np.nanmean)
    def buffer_size(self, value: int) -> None:
        """Record the size of the trajectory buffer.

        :param value: The size of the trajectory buffer.
        """

    @define_epoch_stats(np.nanmean, input_name='time')
    @define_epoch_stats(np.nanmean, input_name='percent')
    def time_sampling_from_buffer(self, time: float, percent: float) -> None:
        """Record the total time it took the learner to sample from the buffer + relative per to total update time.

        :param time: The absolute time it took for the computation.
        :param percent: The relative percentage this computation took w.r.t. to one update.
        """

    @define_epoch_stats(np.nanmean)
    def buffer_avg_pick_per_transition(self, value: int) -> None:
        """Record the cumulative moving average of the picks per transition of the buffer.

        :param value: The cumulative moving average of the number of times a single transitions is sampled from the
            trajectory buffer.
        """
