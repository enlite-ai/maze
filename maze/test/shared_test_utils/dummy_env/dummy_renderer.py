"""
Implements dummy renderer for dummy environment.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.rendering.renderer import Renderer


class DummyRenderer(Renderer):
    """
    Dummy renderer for dummy environment. Doesn't render anything.
    """

    def render(self, maze_state: MazeStateType, maze_action: Optional[MazeActionType], events: StepEventLog, **kwargs) \
            -> None:
        """
        Doesn't render anything.
        """
        pass


class DummyMatplotlibRenderer(Renderer):
    """Dummy matplotlib based rendering class.
    """

    def render(self, maze_state: MazeStateType, maze_action: Optional[MazeActionType], events: StepEventLog, **kwargs) \
            -> None:
        """
        Renders a random integer array with matplotlib.
        """

        plt.figure('DummyEnv', figsize=(8, 4))
        plt.clf()
        plt.imshow(np.random.random_integers(0, 256, size=(64, 64, 3)))
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
