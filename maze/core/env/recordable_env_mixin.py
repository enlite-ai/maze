"""Interface for direct access to current MazeState and MazeAction objects."""

from abc import ABC, abstractmethod

from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.rendering.renderer import Renderer


class RecordableEnvMixin(ABC):
    """
    This interface provides a standard way of exposing internal MazeState and MazeAction objects for trajectory
    data recording.
    """

    @abstractmethod
    def get_maze_state(self) -> MazeStateType:
        """Return current state of the environment.

        :return: Current environment state object.
        """
        raise NotImplementedError

    @abstractmethod
    def get_maze_action(self) -> MazeActionType:
        """Return the last MazeAction taken in the environment

        :return: Last MazeAction object.
        """
        raise NotImplementedError

    @abstractmethod
    def get_episode_id(self) -> str:
        """Get ID of the current episode. Usually a UUID converted to a string, but can be a custom string as well.

        :return: Episode ID string
        """
        raise NotImplementedError

    @abstractmethod
    def get_renderer(self) -> Renderer:
        """Return renderer that can be used to render the recorded trajectory data.

        :return: Renderer instance.
        """
        raise NotImplementedError
