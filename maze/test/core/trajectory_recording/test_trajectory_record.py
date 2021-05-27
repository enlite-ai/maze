from typing import List

import numpy as np

from maze.core.env.structured_env import ActorID
from maze.core.trajectory_recording.records.spaces_record import SpacesRecord
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord


def _mock_space_record(value: int):
    substep_record = SpacesRecord(
        actor_id=ActorID(0, 0),
        observation=dict(observation=np.array(value)),
        action=dict(action=np.array(value)),
        reward=value,
        done=value > 0
    )

    return StructuredSpacesRecord(substep_records=[substep_record])


def _mock_trajectory_record(id: int, values: List[int]):
    t = SpacesTrajectoryRecord(id=id)
    t.step_records = [_mock_space_record(val) for val in values]
    return t


def test_trajectory_stacking():
    t1 = _mock_trajectory_record(1, [11, 12, 13])
    t2 = _mock_trajectory_record(2, [21, 22, 23])
    t3 = _mock_trajectory_record(3, [31, 32, 33])

    # (1) Stack all trajectories into one, keeping the time dimension
    stacked_trajectory = SpacesTrajectoryRecord.stack_trajectories([t1, t2, t3])

    # Step records should be stacked, keeping the time dimension intact
    assert np.all(stacked_trajectory.step_records[0].observations_dict[0]['observation'] == [11, 21, 31])
    assert np.all(stacked_trajectory.step_records[1].observations_dict[0]['observation'] == [12, 22, 32])
    assert np.all(stacked_trajectory.step_records[2].observations_dict[0]['observation'] == [13, 23, 33])

    # IDs should be stacked as well
    assert np.all(stacked_trajectory.id == [1, 2, 3])

    # (2) Stack up the (already stacked) trajectory into one single stacked step record
    stacked_record = stacked_trajectory.stack()
    expected_obs_val = [
        [11, 21, 31],
        [12, 22, 32],
        [13, 23, 33],
    ]
    assert np.all(stacked_record.observations_dict[0]['observation'] == expected_obs_val)
