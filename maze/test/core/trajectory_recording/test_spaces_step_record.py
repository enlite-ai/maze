from typing import List, Union

import numpy as np
import torch

from maze.core.env.structured_env import ActorID
from maze.core.trajectory_recording.records.spaces_record import SpacesRecord
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord


def _mock_spaces_record(
        actor_id: ActorID,
        keys: List[str],
        value: Union[int, List[int]],
        reward: int = 1,
        done: bool = False):

    return SpacesRecord(
        actor_id=actor_id,
        observation={k: np.array(value) for k in keys},
        action={"action": np.array(value)},
        reward=reward,
        done=done
    )


def _mock_structured_spaces_record(step_no: int, done: bool = False):
    return StructuredSpacesRecord(substep_records=[
        _mock_spaces_record(actor_id=ActorID(0, 0), keys=["x", "y"], value=[step_no * 10, step_no * 10], reward=step_no),
        _mock_spaces_record(actor_id=ActorID(1, 0), keys=["z"], value=[step_no * 10 + 1], reward=step_no, done=done),
    ])


def test_record_stacking():
    r1 = _mock_structured_spaces_record(1)
    r2 = _mock_structured_spaces_record(2)
    r3 = _mock_structured_spaces_record(3, done=True)

    stacked = StructuredSpacesRecord.stack_records([r1, r2, r3])

    # Check that the observations are stacked as expected

    expected_observations = {
        0: dict(
            x=np.array([[10, 10], [20, 20], [30, 30]]),
            y=np.array([[10, 10], [20, 20], [30, 30]])
        ),
        1: dict(
            z=np.array([[11], [21], [31]])
        )
    }

    for step_key in [0, 1]:
        for obs_key, exp_value in expected_observations[step_key].items():
            assert np.all(stacked.observations_dict[step_key][obs_key] == exp_value)

    # Check a couple of other values

    assert np.all(stacked.rewards_dict[0] == [1, 2, 3])
    assert np.all(stacked.dones_dict[1] == [False, False, True])
    assert stacked.actions_dict[0]["action"].shape == (3, 2)


def test_record_conversion():
    r = _mock_structured_spaces_record(1)

    r.to_torch("cpu")
    for step_key in [0, 1]:
        for value in r.observations_dict[step_key].values():
            assert isinstance(value, torch.Tensor)
        for value in r.actions_dict[step_key].values():
            assert isinstance(value, torch.Tensor)

    r.to_numpy()
    for step_key in [0, 1]:
        for value in r.observations_dict[step_key].values():
            assert isinstance(value, np.ndarray)
        for value in r.actions_dict[step_key].values():
            assert isinstance(value, np.ndarray)
