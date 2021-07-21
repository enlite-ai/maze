"""Holds methods for preprocessing the trajectories, before passing them through the env and storing them in the
    InMemoryDataset"""
import dataclasses
from abc import abstractmethod
from typing import List, Optional

from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.trajectory_recording.records.state_record import StateRecord
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.records.trajectory_record import TrajectoryRecord


class TrajectoryProcessor:
    """Interface for processing individual trajectories."""

    @abstractmethod
    def pre_process(self, trajectory: TrajectoryRecord) -> TrajectoryRecord:
        """Preprocess a given trajectory before passing it through the wrapper stack.

        :param trajectory: The trajectory to preprocess.
        :return: The preprocessed trajectory method.
        """

    @staticmethod
    def convert_trajectory_with_env(trajectory: TrajectoryRecord, conversion_env: Optional[MazeEnv]) \
            -> List[StructuredSpacesRecord]:
        """Convert an episode trajectory record into an array of observations and actions using the given env.

        :param trajectory: Episode record to load
        :param conversion_env: Env to use for conversion of MazeStates and MazeActions into observations and actions.
                               Required only if state records are being loaded (i.e. conversion to raw actions and
                               observations is needed).
        :return: Loaded observations and actions. I.e., a tuple (observation_list, action_list). Each of the
                 lists contains observation/action dictionaries, with keys corresponding to IDs of structured
                 sub-steps. (I.e., the dictionary will have just one entry for non-structured scenarios.)
        """
        step_records = []

        for step_id, step_record in enumerate(trajectory.step_records):

            # Process and convert in case we are dealing with state records (otherwise no conversion needed)
            if isinstance(step_record, StateRecord):
                assert conversion_env is not None, "when conversion from Maze states is needed, conversion env " \
                                                   "needs to be present."

                # Drop incomplete records (e.g. at the end of episode)
                if step_record.maze_state is None or step_record.maze_action is None:
                    continue
                # Convert to spaces
                step_record = StructuredSpacesRecord.converted_from(step_record, conversion_env=conversion_env,
                                                                    first_step_in_episode=step_id == 0)

            step_records.append(step_record)

        return step_records

    def process(self, trajectory: TrajectoryRecord, conversion_env: Optional[MazeEnv]) \
            -> List[StructuredSpacesRecord]:
        """Convert an individual trajectory, by calling the pre_processing method followed by the
        convert_trajectory_with_env

        :param trajectory: Episode record to load
        :param conversion_env: Env to use for conversion of MazeStates and MazeActions into observations and actions.
                               Required only if state records are being loaded (i.e. conversion to raw actions and
                               observations is needed).
        :return: Loaded observations and actions. I.e., a tuple (observation_list, action_list). Each of the
                 lists contains observation/action dictionaries, with keys corresponding to IDs of structured
                 sub-steps. (I.e., the dictionary will have just one entry for non-structured scenarios.)
        """

        pre_processed_trajectories = self.pre_process(trajectory)
        env_processed_trajectories = self.convert_trajectory_with_env(pre_processed_trajectories, conversion_env)
        return env_processed_trajectories


class IdentityTrajectoryProcessor(TrajectoryProcessor):
    """Identity processing method"""

    @override(TrajectoryProcessor)
    def pre_process(self, trajectory: TrajectoryRecord) -> TrajectoryRecord:
        """Implementation of :class:`~maze.core.trajectory_recording.datasets.trajectory_preprocessing_methods.TrajectoryProcessor` interface.
        """
        return trajectory


@dataclasses.dataclass
class DeadEndClippingTrajectoryProcessor(TrajectoryProcessor):
    """Implementation of the dead-end-clipping preprocessor. That is for each trajectory the last k states should be
    clipped iff the env is done in the last state."""
    clip_k: int

    @override(TrajectoryProcessor)
    def pre_process(self, trajectory: TrajectoryRecord) -> TrajectoryRecord:
        """Implementation of :class:`~maze.core.trajectory_recording.datasets.trajectory_preprocessing_methods.TrajectoryProcessor` interface.
        """
        last_record = trajectory.step_records[-1]
        if isinstance(last_record, StateRecord):
            if last_record.maze_state is None or last_record.maze_action is None:
                trajectory.step_records = trajectory.step_records[:-1]
            is_done = trajectory.step_records[-1].done
            info = trajectory.step_records[-1].info
        elif isinstance(last_record, StructuredSpacesRecord):
            if last_record.observations is None or last_record.observations is [] or last_record.actions is [] \
                    or last_record.actions is None:
                trajectory.step_records = trajectory.step_records[:-1]
            assert not trajectory.step_records[-1].is_batched(), "cannot determine done state for batched trajectory."
            is_done = trajectory.step_records[-1].is_done()
            info = trajectory.step_records[-1].substep_records[0].info
        else:
            raise ValueError(f'Unrecognized trajectory encountered -> type: {type(last_record)}, value: {last_record}')

        if len(trajectory) > self.clip_k and is_done and \
                (info is None or 'TimeLimit.truncated' not in info):
            trajectory.step_records = trajectory.step_records[:-self.clip_k]

        return trajectory
