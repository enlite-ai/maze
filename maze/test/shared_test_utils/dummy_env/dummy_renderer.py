"""
Implements dummy renderer for dummy environment.
"""

from typing import Optional

from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.rendering.renderer import Renderer


class DummyRenderer(Renderer):
    """
    Dummy renderer for dummy environment. Doesn't render anything.
    """

    def render(self, maze_state: MazeStateType, maze_action: Optional[MazeActionType], events: StepEventLog, **kwargs) -> None:
        """
        Doesn't render anything.
        """
        pass
