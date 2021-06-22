"""IMPALA Events"""

from abc import ABC

import numpy as np

from maze.core.log_stats.event_decorators import define_epoch_stats


class ImpalaEvents(ABC):
    """Events specific for the impala algorithm, in order to record and analyse it's behaviour in more detail"""

    @define_epoch_stats(np.nanmean, input_name='time')
    @define_epoch_stats(np.nanmean, input_name='percent')
    def time_dequeuing_actors(self, time: float, percent: float):
        """Record the time it took to dequeue the actors output from the synced queue + relative per to total update
        time

        :param time: the absolute time it took for the computation
        :param percent: the relative percentage this computation took w.r.t. to one update
        """

    @define_epoch_stats(np.nanmean, input_name='time')
    @define_epoch_stats(np.nanmean, input_name='percent')
    def time_collecting_actors(self, time: float, percent: float):
        """Record the total time it took the learner to collect the actors output + relative per to total update time

        :param time: the absolute time it took for the computation
        :param percent: the relative percentage this computation took w.r.t. to one update
        """

    @define_epoch_stats(np.nanmean, input_name='time')
    @define_epoch_stats(np.nanmean, input_name='percent')
    def time_learner_rollout(self, time: float, percent: float):
        """Record the total time it took the learner to compute the logits on the agents output
            + relative per to total update time

        :param time: the absolute time it took for the computation
        :param percent: the relative percentage this computation took w.r.t. to one update
        """

    @define_epoch_stats(np.nanmean, input_name='time')
    @define_epoch_stats(np.nanmean, input_name='percent')
    def time_loss_computation(self, time: float, percent: float):
        """Record the total time it took the learner compute the loss + relative per to total update time

        :param time: the absolute time it took for the computation
        :param percent: the relative percentage this computation took w.r.t. to one update
        """

    @define_epoch_stats(np.nanmean, input_name='time')
    @define_epoch_stats(np.nanmean, input_name='percent')
    def time_backprob(self, time: float, percent: float):
        """Record the total time it took the learner to backprob the loss + relative per to total update time

        :param time: the absolute time it took for the computation
        :param percent: the relative percentage this computation took w.r.t. to one update
        """

    @define_epoch_stats(np.nanmean, input_name='before')
    @define_epoch_stats(np.nanmean, input_name='after')
    def estimated_queue_sizes(self, before: int, after: int):
        """Record the estimated queue size before and after the collection of the actors output

        :param before: the estimated queue size before collection
        :param after: the estimated queue size after collection
        """
