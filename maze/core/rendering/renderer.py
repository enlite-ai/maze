"""Renderer is the main interface for renderer classes that render current state of an env."""

from abc import abstractmethod, ABC
from typing import List, Optional

from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.rendering.renderer_args import RendererArg


class Renderer(ABC):
    """Interface for renderers of individual environments.

    Renders state of one particular step -- based only on current state.
    """

    @staticmethod
    def arguments() -> List[RendererArg]:
        """List the additional arguments that the renderer supports (beyond maze_state and maze_action), if any.

        Exposing available argument options like this makes it possible to create appropriate user controls
        when controlling the renderer in interactive settings (e.g., using widgets in a Jupyter Notebook).

        Note:

        Note that the types and names of arguments returned are expected to be the same across all possible env
        configurations. What can change are the available values, which then are expected to stay the same for
        a whole episode.

        Example of this are drivers in a vehicle routing env. If you would like to display a detail of the driver,
        a driver_id argument can be exposed. It will always be named `driver_id` and be of the same type, but
        across different episodes, the number of drivers (i.e. allowed range of the argument) might differ. It
        will always stay fixed during a whole episode though.

        :return: List of renderer argument objects.
        """
        return []

    @abstractmethod
    def render(self, maze_state: MazeStateType, maze_action: Optional[MazeActionType], events: StepEventLog,
               **kwargs) -> None:
        """Render the current state as a matplotlib figure.

        Note that the maze_action is optional -- it is None for the last (terminal) state in the episode!

        :param maze_state: State to render.
        :param maze_action: MazeAction to render. Should be the MazeAction derived from the state to render
                            (provided above)
        :param events: Events dispatched by the env during the last step (i.e. when the given state was produced)
        :param kwargs: Any additional arguments that the renderer accepts and exposes
        """
        raise NotImplementedError
