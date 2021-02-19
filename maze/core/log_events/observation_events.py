"""Events and plotting functions for observations """
from abc import ABC

from maze.core.log_stats.event_decorators import define_episode_stats, define_step_stats, define_epoch_stats, \
    define_stats_grouping
from maze.core.log_stats.reducer_functions import histogram


class ObservationEvents(ABC):
    """
    Event topic class with logging statistics based only on observations, therefore applicable to any valid
    reinforcement learning environment.
    """

    @define_epoch_stats(histogram)
    @define_episode_stats(histogram)
    @define_step_stats(None)
    @define_stats_grouping("step_key", "name")
    def observation_processed(self, step_key: str, name: str, value: int):
        """ observation seen and dimensionality of observation space """

    @define_epoch_stats(histogram)
    @define_episode_stats(histogram)
    @define_step_stats(None)
    @define_stats_grouping("step_key", "name")
    def observation_original(self, step_key: str, name: str, value: int):
        """ observation seen and dimensionality of observation space """
