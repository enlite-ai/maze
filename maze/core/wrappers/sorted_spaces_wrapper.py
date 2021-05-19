"""This file contains an env wrapper which sorts the observations and actions as well as their corresponding spaces
    by the name of the key-names."""
from typing import Optional, Tuple, Dict, Union, Any

import gym

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import Wrapper, EnvType


class SortedSpacesWrapper(Wrapper[Union[EnvType, StructuredEnvSpacesMixin]]):
    """This class wraps a given StructuredEnvSpacesMixin env to ensure that all observation- and action-spaces are sorted
    alphabetically. This is required that Maze custom action distributions and observation processing are in line
    with RLLib's internal processing pipeline.
    """

    @property
    @override(StructuredEnvSpacesMixin)
    def action_space(self) -> gym.spaces.Dict:
        """The currently active gym action space.
        """
        return gym.spaces.Dict(sorted(self.env.action_space.spaces.items()))

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_space(self) -> gym.spaces.Dict:
        """Keep this env compatible with the gym interface by returning the
        observation space of the current policy."""
        return gym.spaces.Dict(sorted(self.env.observation_space.spaces.items()))

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType], maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) -> Tuple[Optional[Dict[Union[int, str], Any]],
                                                                               Optional[Dict[Union[int, str], Any]]]:
        """This wrapper does not modify observations and actions."""
        return self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'SortedSpacesWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
