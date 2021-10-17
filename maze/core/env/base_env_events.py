"""Generic event interfaces"""
from abc import ABC

import numpy as np

from maze.core.log_stats.event_decorators import define_episode_stats, define_step_stats, define_epoch_stats, \
    define_stats_grouping


class BaseEnvEvents(ABC):
    """
    Event topic class with logging statistics based only on _reward_, therefore applicable to any valid
    reinforcement learning environment.

    Defined statistics:
    - Cumulative number of total steps
    - Mean epoch reward and standard deviation
    """

    @define_epoch_stats(np.max, input_name="sum", output_name="max")
    @define_epoch_stats(np.min, input_name="sum", output_name="min")
    @define_epoch_stats(np.mean, input_name="sum", output_name="mean")
    @define_epoch_stats(np.std, input_name="sum", output_name="std")
    @define_epoch_stats(sum, input_name="count", output_name="total_step_count", cumulative=True)
    @define_epoch_stats(np.mean, input_name="count", output_name="mean_step_count")
    @define_epoch_stats(np.median, input_name="count", output_name="median_step_count")
    @define_epoch_stats(len, input_name="sum", output_name="episode_count")
    @define_epoch_stats(len, input_name="sum", output_name="total_episode_count", cumulative=True)
    @define_episode_stats(np.sum, output_name="sum")
    @define_episode_stats(len, output_name="count")
    @define_step_stats(sum)
    def reward(self, value: float):
        """reward value for the current step"""

    @define_epoch_stats(np.mean, output_name="mean")
    @define_epoch_stats(np.std, output_name="std")
    @define_epoch_stats(np.min, output_name="min")
    @define_epoch_stats(np.max, output_name="max")
    @define_episode_stats(None)
    @define_step_stats(None)
    @define_stats_grouping("name")
    def kpi(self, name: str, value: float):
        """Event representing a KPI metric (Key Performance Indicator).

        KPI metrics are expected to be calculated at the end of the episode. Only one KPI value
        per KPI name per episode should be recorded.

        :param name: Name of the KPI metric
        :param value: Value of the KPI metric
        """
