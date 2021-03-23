"""Rendering trajectory data in a simple form."""
from typing import Dict, Any

from maze.core.rendering.renderer import Renderer
from maze.core.trajectory_recording.records.trajectory_record import StateTrajectoryRecord


class KeyboardControlledTrajectoryViewer:
    """
    Render trajectory data with the possibility to browse back and forward through the episode steps using keyboard.

    This is the simplest form of interactive rendering of episode trajectory, useful for example when rendering
    the trajectory ad hoc while the environment is still running. For more comfortable rendering of trajectory
    data inside of a Jupyter Notebook, use :class:`~.notebook_trajectory_viewer.NotebookTrajectoryViewer`.

    Note:

    The keyboard controls might not work well when run outside of terminal.

    If running this through PyCharm, the "Emulate terminal in output console" options in Run/Debug configurations
    needs to be set to true, otherwise the keys will not be picked up correctly and this run loop will crash.

    Also, the console needs to be active dor the keys to be picked up.
    """

    def __init__(self,
                 episode_record: StateTrajectoryRecord,
                 renderer: Renderer,
                 initial_step_index: int = 0,
                 renderer_kwargs: Dict[str, Any] = None):
        self.episode_record = episode_record
        self.renderer = renderer
        self.step_index = initial_step_index
        self.renderer_kwargs = renderer_kwargs if renderer_kwargs is not None else {}

    def render(self):
        """
        Run the interactive rendering loop. Waits for user input (right or left arrow), updates the step index
        accordingly and triggers the redraw through the renderer.
        """
        from getkey import getkey, keys

        self._print_step_and_render()
        while True:
            key = getkey()
            if key == keys.LEFT:
                self._render_previous_step()
            elif key == keys.RIGHT:
                self._render_next_step()
            elif key == keys.ESC:
                print()
                return
            else:
                print("Invalid key. Press right/left arrow or Esc.", end="\r")

    def _render_next_step(self):
        if self.step_index == len(self.episode_record.step_records) - 1:
            return

        self.step_index += 1
        self._print_step_and_render()

    def _render_previous_step(self):
        if self.step_index == 0:
            return

        self.step_index -= 1
        self._print_step_and_render()

    def _print_step_and_render(self):
        if self.step_index == 0 == len(self.episode_record.step_records) - 1:
            suffix = "(the only step in episode)"
        elif self.step_index == 0:
            suffix = "(beginning of episode)    "
        elif self.step_index == len(self.episode_record.step_records) - 1:
            suffix = "(end of episode)          "
        else:
            suffix = "                          "
        print(f"Current step: {self.step_index} {suffix}", end="\r")

        step_record = self.episode_record.step_records[self.step_index]
        self.renderer.render(maze_state=step_record.maze_state, maze_action=step_record.maze_action,
                             events=step_record.step_event_log, **self.renderer_kwargs)
