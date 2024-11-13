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

    # The decorators allow specifying which reduction method to use. For string values, this requires special handling,
    # as methods like min or max cannot be applied to strings. The lambda function addresses this by returning the
    # input value unchanged if it's a string (acting as an identity function), or applying the specified reduction
    # otherwise.
    @define_epoch_stats(lambda x: x if any(isinstance(i, str) for i in x) else np.mean(x), output_name="mean")
    @define_epoch_stats(lambda x: x if any(isinstance(i, str) for i in x) else np.std(x), output_name="std")
    @define_epoch_stats(lambda x: x if any(isinstance(i, str) for i in x) else np.min(x), output_name="min")
    @define_epoch_stats(lambda x: x if any(isinstance(i, str) for i in x) else np.max(x), output_name="max")
    @define_episode_stats(None)
    @define_step_stats(None)
    @define_stats_grouping("name")
    def kpi(self, name: str, value: float | str):
        """Event representing a KPI metric (Key Performance Indicator).

        KPI metrics are expected to be calculated at the end of the episode. Only one KPI value
        per KPI name per episode should be recorded.

        :param name: Name of the KPI metric
        :param value: Value of the KPI metric
        """

    @define_epoch_stats(sum)
    @define_episode_stats(sum)
    @define_step_stats(sum)
    def test_event(self, value):
        """Test event, can be used for testing purposes"""
