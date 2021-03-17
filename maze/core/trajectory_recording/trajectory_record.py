"""Episode record is the main unit of trajectory data recording."""

from typing import List, Optional, Any, TypeVar, Generic

import numpy as np

from maze.core.rendering.renderer import Renderer
from maze.core.trajectory_recording.spaces_step_record import SpacesStepRecord
from maze.core.trajectory_recording.state_step_record import StateStepRecord

StepRecordType = TypeVar('StepRecordType', SpacesStepRecord, StateStepRecord)


class TrajectoryRecord(Generic[StepRecordType]):
    def __init__(self, id: Any):
        self.id = id
        self.step_records: List[StepRecordType] = []

    def __len__(self):
        return len(self.step_records)

    def append(self, step_record: StepRecordType) -> 'TrajectoryRecord':
        self.step_records.append(step_record)
        return self

    def extend(self, trajectory_record: 'TrajectoryRecord') -> 'TrajectoryRecord':
        self.step_records.extend(trajectory_record.step_records)
        return self

    def __repr__(self):
        return f"\nTrajectory record:\n" \
               f"   ID: {self.id}\n" \
               f"   Length: {len(self)}"


class StateTrajectoryRecord(TrajectoryRecord[StateStepRecord]):
    """Records and keeps trajectory record data for a complete episode.

    :param id: ID of the episode. Can be used to link trajectory data from event logs and other sources.
    :param renderer: Where available, the renderer object should be associated to the episode record. This ensures
       correct configuration of the renderer (with respect to env configuration for this episode), and
       makes it easier to instantiate the correct renderer for displaying the trajectory data.
    """

    def __init__(self, id: Any, renderer: Optional[Renderer] = None):
        super().__init__(id)
        self.renderer = renderer


class SpacesTrajectoryRecord(TrajectoryRecord[SpacesStepRecord]):

    def stack(self) -> SpacesStepRecord:
        assert all([isinstance(record, SpacesStepRecord) for record in self.step_records]), \
            "stacking supported by records of spaces only"
        return SpacesStepRecord.stack_records(self.step_records)

    @classmethod
    def stack_trajectories(cls, trajectories: List['SpacesTrajectoryRecord']) -> 'SpacesTrajectoryRecord':
        assert len(set([len(t) for t in trajectories])) == 1, "all trajectories must have the same length"

        stacked_trajectory = SpacesTrajectoryRecord(id=np.stack([trajectory.id for trajectory in trajectories]))
        step_records_in_time = list(zip(*[t.step_records for t in trajectories]))
        stacked_trajectory.step_records = [SpacesStepRecord.stack_records(list(recs)) for recs in step_records_in_time]
        return stacked_trajectory

    def is_done(self):
        return list(self.step_records[-1].dones.values())[-1] if len(self) > 0 else False

    @property
    def actions(self):
        return [step_record.actions for step_record in self.step_records]

    def total_reward(self):
        total_reward = 0
        for record in self.step_records:
            for reward in record.rewards.values():
                total_reward += reward
        return total_reward