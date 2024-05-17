"""File holding utility methods for the dataset"""
from typing import Tuple, Dict

from maze.core.env.maze_env import MazeEnv
from maze.core.trajectory_recording.records.state_record import StateRecord
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.records.trajectory_record import TrajectoryRecord


def retrieve_done_and_last_info(trajectory: TrajectoryRecord) -> Tuple[bool, Dict]:
    """Helper method to retrieve the information on how the given trajectory ended.

    :param trajectory: Episode record to load.
    :return: A bool indicating whether the trajectory ended with a done and the last info dict.
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
        is_done = trajectory.step_records[-1].is_done()
        info = trajectory.step_records[-1].substep_records[-1].info
    else:
        raise ValueError(f'Unrecognized trajectory encountered -> type: {type(last_record)}, value: {last_record}')
    return is_done, info


def retrieve_done_info(trajectory: TrajectoryRecord) -> Tuple[bool, bool, Dict]:
    """Helper method to retrieve the information on how the given trajectory ended.

    :param trajectory: Episode record to load.
    :return: A bool indicating whether the trajectory ended with a done by termination, done by timelimit truncation
            and the last info dict.
    """
    done, info = retrieve_done_and_last_info(trajectory)
    done_terminated, done_truncated = MazeEnv.get_done_info(done, info)
    assert not (done_terminated and done_truncated)
    return done_terminated, done_truncated, info