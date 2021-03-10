"""Episode record is the main unit of trajectory data recording."""

from typing import List, Optional, Union, Any

import numpy as np

from maze.core.rendering.renderer import Renderer
from maze.core.trajectory_recorder.spaces_step_record import SpacesStepRecord
from maze.core.trajectory_recorder.state_step_record import StateStepRecord


class TrajectoryRecord:
    """Records and keeps trajectory record data for a complete episode.

    :param trajectory_id: ID of the episode. Can be used to link trajectory data from event logs and other sources.
    :param renderer: Where available, the renderer object should be associated to the episode record. This ensures
       correct configuration of the renderer (with respect to env configuration for this episode), and
       makes it easier to instantiate the correct renderer for displaying the trajectory data.
    """

    def __init__(self, trajectory_id: Any, renderer: Optional[Renderer] = None):
        self.trajectory_id = trajectory_id
        self.step_records: List[Union[StateStepRecord, SpacesStepRecord]] = []
        self.renderer = renderer

    def __len__(self):
        return len(self.step_records)

    def append(self, step_record: Union[StateStepRecord, SpacesStepRecord]) -> 'TrajectoryRecord':
        self.step_records.append(step_record)
        return self

    def extend(self, trajectory_record: 'TrajectoryRecord') -> 'TrajectoryRecord':
        self.step_records.extend(trajectory_record.step_records)
        return self

    @classmethod
    def stack_trajectories(cls, trajectories: List['TrajectoryRecord']) -> 'TrajectoryRecord':
        assert len(set([len(t) for t in trajectories])) == 1, "all trajectories must have the same length"

        stacked_trajectory = TrajectoryRecord(
            trajectory_id=np.stack([trajectory.trajectory_id for trajectory in trajectories])
        )
        step_records_in_time = list(zip(*[t.step_records for t in trajectories]))
        stacked_trajectory.step_records = [SpacesStepRecord.stack_records(list(recs)) for recs in step_records_in_time]
        return stacked_trajectory

    def stack(self) -> SpacesStepRecord:
        assert all([isinstance(record, SpacesStepRecord) for record in self.step_records]), \
            "stacking supported by records of spaces only"
        return SpacesStepRecord.stack_records(self.step_records)

    def __repr__(self):
        return f"\nTrajectory record:\n" \
               f"   ID: {self.trajectory_id}\n" \
               f"   Length: {len(self)}"
