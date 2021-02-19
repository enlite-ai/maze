"""Episode record is the main unit of trajectory data recording."""

from typing import List, Optional

from maze.core.rendering.renderer import Renderer
from maze.core.trajectory_recorder.step_record import StepRecord


class EpisodeRecord:
    """Records and keeps trajectory record data for a complete episode.

    :param episode_id: ID of the episode. Can be used to link trajectory data from event logs and other sources.
    :param renderer: Where available, the renderer object should be associated to the episode record. This ensures
       correct configuration of the renderer (with respect to env configuration for this episode), and
       makes it easier to instantiate the correct renderer for displaying the trajectory data.
    """

    def __init__(self, episode_id: str, renderer: Optional[Renderer] = None):
        self.episode_id = episode_id
        self.step_records: List[StepRecord] = []
        self.renderer = renderer
