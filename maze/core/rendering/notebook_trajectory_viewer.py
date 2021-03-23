"""Displaying trajectory data as interactive plots in Jupyter Notebooks."""

from maze.core.rendering.renderer_args import RendererArg, IntRangeArg
from maze.core.trajectory_recording.records.trajectory_record import StateTrajectoryRecord


class NotebookTrajectoryViewer:
    """Trajectory viewer for Jupyter Notebooks, built using ipython widgets.

    Displays trajectory data for the given episode as an interactive view, where the step ID and any additional
    arguments exposed by the renderer can be interactively set. The data is rendered using the renderer recorded
    in the episode record.

    :param episode_record: Trajectory data to render.
    """

    def __init__(self, episode_record: StateTrajectoryRecord):
        self.episode_record = episode_record
        self.renderer = episode_record.renderer

    def build(self) -> None:
        """Build and show the interactive widgets.

        Builds all the widgets (one for step ID, then one for each additional argument accepted by the renderer)
        and activates them using the interact function. Expected to be called from a cell in a Jupyter notebook.
        """
        from ipywidgets import interact

        step_count = len(self.episode_record.step_records)
        step_id_argument: RendererArg = IntRangeArg(name="step_id", title="Step ID",
                                                    min_value=0, max_value=(step_count - 1))
        arguments = [step_id_argument] + self.renderer.arguments()
        widgets = {arg.name: arg.create_widget() for arg in arguments}
        interact(self.render, **widgets)

    def render(self, step_id, **kwargs) -> None:
        """Render the view for the given step ID, with the given additional parameters.

        Usually, this method is not called directly -- it is expected to be called by ipython widgets.

        :param step_id: ID of the step to display
        :param kwargs: Any additional arguments the renderer accepts
        """
        assert 0 <= step_id < len(self.episode_record.step_records)
        step_record = self.episode_record.step_records[step_id]
        self.renderer.render(step_record.maze_state, step_record.maze_action, step_record.step_event_log, **kwargs)
