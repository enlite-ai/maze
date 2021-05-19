""" Contains an example showing how to visualize processed observations (network input) with tensorboard. """
from abc import ABC
import warnings
from typing import TypeVar, Union, Any, Tuple, Dict, Optional, List, Callable

import matplotlib.pyplot as plt
import numpy as np
from maze.core.annotations import override, unused
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env import StructuredEnv
from maze.core.log_stats.event_decorators import define_plot, define_epoch_stats, define_episode_stats, \
    define_step_stats, define_stats_grouping
from maze.core.log_stats.reducer_functions import histogram
from maze.core.utils.factory import Factory
from maze.core.wrappers.wrapper import Wrapper


def histogram_plot(value: List[np.ndarray], **kwargs) -> None:
    """Plots histograms of all observations (default behaviour).

    :param value: A list of observation arrays.
    :param kwargs: Additional plotting relevant arguments.
    """
    unused(kwargs)

    flat_values = np.stack(value).flatten()

    fig = plt.figure(figsize=(7, 5))
    plt.hist(flat_values)
    return fig


class ObservationVisualizationEvents(ABC):
    """Event topic class with logging statistics based only on observations, therefore applicable to any valid
    reinforcement learning environment.
    """

    # create_figure_function will be overwritten in the
    # ObservationVisualizationWrapper with a custom plotting function
    @define_plot(create_figure_function=histogram_plot, input_name=None)
    @define_epoch_stats(histogram)
    @define_episode_stats(histogram)
    @define_step_stats(None)
    @define_stats_grouping("step_key", "name")
    def observation_to_visualize(self, step_key: str, name: str, value: int):
        """ observation to be visualized """


class ObservationVisualizationWrapper(Wrapper[MazeEnv]):
    """An observation visualization wrapper allows to apply custom observation visualization functions
    which are then shown in Tensorboard.

    :param env: The environment to wrap.
    :param plot_function: The custom matplotlib plotting function.
    """

    T = TypeVar("T")

    def __init__(self, env: MazeEnv, plot_function: Optional[str]):
        """Avoid calling this constructor directly, use :method:`wrap` instead."""
        super().__init__(env)

        # create event topics
        self.observation_events = self.core_env.context.event_service.create_event_topic(ObservationVisualizationEvents)

        # update plot function
        if plot_function is not None:
            function = Factory(Callable).type_from_name(plot_function)
            ObservationVisualizationEvents.observation_to_visualize.tensorboard_render_figure_dict[None] = function

    @override(BaseEnv)
    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, Dict[Any, Any]]:
        """Triggers logging events for observations, actions and reward.
        """

        # get identifier of current sub step
        substep_id, _ = self.env.actor_id() if isinstance(self.env, StructuredEnv) else (None, None)
        substep_name = f"step_key_{substep_id}" if substep_id is not None else None

        # take wrapped env step
        obs, rew, done, info = self.env.step(action)

        # log processed observations
        for observation_name, observation_value in obs.items():
            self.observation_events.observation_to_visualize(
                step_key=substep_name, name=observation_name, value=observation_value)

        return obs, rew, done, info

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Keep both actions and observation the same."""
        return self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'ObservationVisualizationWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        warnings.warn("Try to avoid wrappers such as the 'ObservationVisualizationWrapper'"
                      "when working with simulated envs to reduce overhead.")
        self.env.clone_from(env)
