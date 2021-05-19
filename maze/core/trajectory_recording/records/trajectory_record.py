"""Episode record is the main unit of trajectory data recording."""

from typing import List, Optional, Any, TypeVar, Generic, Dict, Union

import numpy as np

from maze.core.env.action_conversion import ActionType
from maze.core.env.observation_conversion import ObservationType
from maze.core.rendering.renderer import Renderer
from maze.core.trajectory_recording.records.state_record import StateRecord
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord, StepKeyType

StepRecordType = TypeVar('StepRecordType', StructuredSpacesRecord, StateRecord)


class TrajectoryRecord(Generic[StepRecordType]):
    """Common functionality of trajectory records.

    :param id: ID of the trajectory. Can be a string ID or any other data (like environment seeding information).
    """

    def __init__(self, id: Any):
        self.id = id
        self.step_records: List[StepRecordType] = []

    def __len__(self):
        """Length of the trajectory"""
        return len(self.step_records)

    def append(self, step_record: StepRecordType) -> None:
        """Append a single step record."""
        self.step_records.append(step_record)

    def extend(self, trajectory_record: 'TrajectoryRecord') -> None:
        """Extend this trajectory with another trajectory."""
        self.step_records.extend(trajectory_record.step_records)

    def __repr__(self):
        return f"\nTrajectory record:\n" \
               f"   ID: {self.id}\n" \
               f"   Length: {len(self)}"


class StateTrajectoryRecord(TrajectoryRecord[StateRecord]):
    """Holds state record data (i.e., Maze states and actions, independent of the current
     action and observation space format) for a complete episode.

    :param id: ID of the episode. Can be used to link trajectory data from event logs and other sources.
    :param renderer: Where available, the renderer object should be associated to the episode record. This ensures
       correct configuration of the renderer (with respect to env configuration for this episode), and
       makes it easier to instantiate the correct renderer for displaying the trajectory data.
    """

    def __init__(self, id: Any, renderer: Optional[Renderer] = None):
        super().__init__(id)
        self.renderer = renderer


class SpacesTrajectoryRecord(TrajectoryRecord[StructuredSpacesRecord]):
    """Holds structured spaces records (i.e., raw actions and observations recorded during a rollout). """

    def stack(self) -> StructuredSpacesRecord:
        """Stack the whole trajectory into a single structured spaces record.

        Useful for processing whole fixed-length trajectories in a single batch.
        """
        return StructuredSpacesRecord.stack_records(self.step_records)

    @classmethod
    def stack_trajectories(cls, trajectories: List['SpacesTrajectoryRecord']) -> 'SpacesTrajectoryRecord':
        """Stack multiple trajectories, keeping the time dimension intact.

        All the trajectories should be of the same length. The resulting trajectory will have the same number of steps,
        each being a stack of the corresponding steps of the input trajectories.

        :param trajectories: Trajectories to stack.
        :return: Trajectory record of the same lenght, consisting of stacked structured spaces records.
        """
        assert len(set([len(t) for t in trajectories])) == 1, "all trajectories must have the same length"

        stacked_trajectory = SpacesTrajectoryRecord(id=np.stack([trajectory.id for trajectory in trajectories]))
        step_records_in_time = list(zip(*[t.step_records for t in trajectories]))
        stacked_trajectory.step_records = [StructuredSpacesRecord.stack_records(list(recs)) for recs in
                                           step_records_in_time]
        return stacked_trajectory

    @property
    def actions(self) -> List[Dict[StepKeyType, ActionType]]:
        """Convenience access to all structured action dicts from this trajectory."""
        return [step_record.actions_dict for step_record in self.step_records]

    @property
    def observations(self) -> List[Dict[StepKeyType, ObservationType]]:
        """Convenience access to all structured observation dicts from this trajectory."""
        return [step_record.observations_dict for step_record in self.step_records]

    @property
    def rewards(self) -> List[Dict[StepKeyType, Union[float, np.ndarray]]]:
        """Convenience access to all structured reward dicts from this trajectory."""
        return [step_record.rewards_dict for step_record in self.step_records]

    def is_done(self) -> bool:
        """Convenience method for checking whether the end of this trajectory represents also the end of an episode."""
        if len(self) == 0:
            return False

        assert not self.step_records[-1].is_batched(), "cannot determine done state for batched trajectory."
        return self.step_records[-1].is_done()

    def total_reward(self):
        """Convenience method for calculating the total reward of a given trajectory."""
        if len(self) == 0:
            return 0

        assert not self.step_records[-1].is_batched(), "cannot determine total reward for a batched trajectory."
        total_reward = 0
        for record in self.step_records:
            total_reward += sum(record.rewards_dict.values())
        return total_reward
