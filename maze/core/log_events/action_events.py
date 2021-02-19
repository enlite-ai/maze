"""Events and plotting functions for discrete action spaces"""
from abc import ABC

from maze.core.log_events.log_create_figure_functions import create_categorical_plot, create_violin_distribution
from maze.core.log_stats.event_decorators import define_episode_stats, define_step_stats, define_epoch_stats, \
    define_stats_grouping, define_plot
from maze.core.log_stats.reducer_functions import histogram


class DiscreteActionEvents(ABC):
    """
    Event topic class with logging statistics based only on discrete (categorical) actions,
    therefore applicable to any valid reinforcement learning environment.
    """

    @define_plot(create_figure_function=create_categorical_plot, input_name=None)
    @define_epoch_stats(histogram)
    @define_episode_stats(histogram)
    @define_step_stats(None)
    @define_stats_grouping("substep", "name")
    def action(self, substep: str, name: str, value: int, action_dim: int):
        """ action taken and dimensionality of discrete action space """


class ContinuousActionEvents(ABC):
    """
    Event topic class with logging statistics based only on continuous actions (box spaces),
    therefore applicable to any valid reinforcement learning environment.
    """

    @define_plot(create_figure_function=create_violin_distribution, input_name=None)
    @define_epoch_stats(histogram)
    @define_episode_stats(histogram)
    @define_step_stats(None)
    @define_stats_grouping("substep", "name")
    def action(self, substep: str, name: str, value: int):
        """ action taken and shape of box action space """
