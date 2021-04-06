"""Contains monitoring events."""
from abc import ABC

import numpy as np

from maze.core.log_events.log_create_figure_functions import create_categorical_plot, create_violin_distribution, \
    create_binary_plot
from maze.core.log_stats.event_decorators import define_episode_stats, define_step_stats, define_epoch_stats, \
    define_stats_grouping, define_plot
from maze.core.log_stats.reducer_functions import histogram


class ActionEvents(ABC):
    """
    Event topic class with logging statistics based only on Gym space actions,
    therefore applicable to any valid reinforcement learning environment.
    """

    @define_plot(create_figure_function=create_binary_plot, input_name=None)
    @define_epoch_stats(histogram)
    @define_episode_stats(histogram)
    @define_step_stats(None)
    @define_stats_grouping("step_key", "name")
    def multi_binary_action(self, step_key: str, name: str, value: int, num_binary_actions: int):
        """ action taken """

    @define_plot(create_figure_function=create_categorical_plot, input_name=None)
    @define_epoch_stats(histogram)
    @define_episode_stats(histogram)
    @define_step_stats(None)
    @define_stats_grouping("step_key", "name")
    def discrete_action(self, step_key: str, name: str, value: int, action_dim: int):
        """ action taken and dimensionality of discrete action space """

    @define_plot(create_figure_function=create_violin_distribution, input_name=None)
    @define_epoch_stats(histogram)
    @define_episode_stats(histogram)
    @define_step_stats(None)
    @define_stats_grouping("step_key", "name")
    def continuous_action(self, step_key: str, name: str, value: int):
        """ action taken and shape of box action space """


class RewardEvents(ABC):
    """
    Event topic class with logging statistics based only on rewards, therefore applicable to any valid
    reinforcement learning environment.
    """

    @define_epoch_stats(np.max, input_name="sum", output_name="max")
    @define_epoch_stats(np.min, input_name="sum", output_name="min")
    @define_epoch_stats(np.mean, input_name="sum", output_name="mean")
    @define_epoch_stats(np.std, input_name="sum", output_name="std")
    @define_episode_stats(np.sum, output_name="sum")
    @define_step_stats(sum)
    @define_stats_grouping("step_key")
    def reward_processed(self, step_key: str, value: float):
        """ reward after being processed by reward wrappers """

    @define_epoch_stats(np.max, input_name="sum", output_name="max")
    @define_epoch_stats(np.min, input_name="sum", output_name="min")
    @define_epoch_stats(np.mean, input_name="sum", output_name="mean")
    @define_epoch_stats(np.std, input_name="sum", output_name="std")
    @define_epoch_stats(sum, input_name="count", output_name="total_step_count", cumulative=True)
    @define_epoch_stats(np.mean, input_name="count", output_name="mean_step_count")
    @define_epoch_stats(np.median, input_name="count", output_name="median_step_count")
    @define_episode_stats(np.sum, output_name="sum")
    @define_episode_stats(len, output_name="count")
    @define_step_stats(sum)
    def reward_original(self, value: float):
        """ original reward returned by the MazeEnv. """


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
        """ observation after being processed by observation wrappers """

    @define_epoch_stats(histogram)
    @define_episode_stats(histogram)
    @define_step_stats(None)
    @define_stats_grouping("step_key", "name")
    def observation_original(self, step_key: str, name: str, value: int):
        """ original MazeEnv observation before being processed be observation wrappers """
