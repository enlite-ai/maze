import numpy as np
import torch

from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord


def test_record_stacking():
    r1 = StructuredSpacesRecord(
        observations={0: dict(x=np.array([10, 10]), y=np.array([10, 10])),
                      1: dict(z=np.array([[11, 11], [11, 11]]))},
        actions={0: dict(action=np.array([10])), 1: dict(action=np.array([11, 11]))},
        rewards={0: 1, 1: 1},
        dones={0: False, 1: False}
    )

    r2 = StructuredSpacesRecord(
        observations={0: dict(x=np.array([20, 20]), y=np.array([20, 20])),
                      1: dict(z=np.array([[21, 21], [21, 21]]))},
        actions={0: dict(action=np.array([20])), 1: dict(action=np.array([21, 21]))},
        rewards={0: 2, 1: 2},
        dones={0: False, 1: False}
    )

    r3 = StructuredSpacesRecord(
        observations={0: dict(x=np.array([30, 30]), y=np.array([30, 30])),
                      1: dict(z=np.array([[31, 31], [31, 31]]))},
        actions={0: dict(action=np.array([30])), 1: dict(action=np.array([31, 31]))},
        rewards={0: 3, 1: 3},
        dones={0: False, 1: True}
    )

    stacked = StructuredSpacesRecord.stack_records([r1, r2, r3])

    # Check that the observations are stacked as expected

    expected_observations = {
        0: dict(
            x=np.array([[10, 10], [20, 20], [30, 30]]),
            y=np.array([[10, 10], [20, 20], [30, 30]])
        ),
        1: dict(
            z=np.array([[[11, 11], [11, 11]],
                        [[21, 21], [21, 21]],
                        [[31, 31], [31, 31]]])
        )
    }

    for step_key in [0, 1]:
        for obs_key, exp_value in expected_observations[step_key].items():
            assert np.all(stacked.observations[step_key][obs_key] == exp_value)

    # Check a couple of other values

    assert np.all(stacked.rewards[0] == [1, 2, 3])
    assert np.all(stacked.dones[1] == [False, False, True])
    assert stacked.actions[1]["action"].shape == (3, 2)


def test_record_conversion():
    r = StructuredSpacesRecord(
        observations={0: dict(x=np.array([10, 10]), y=np.array([10, 10])),
                      1: dict(z=np.array([[11, 11], [11, 11]]))},
        actions={0: dict(action=np.array([10])), 1: dict(action=np.array([11, 11]))},
        rewards={0: 1, 1: 1},
        dones={0: False, 1: False}
    )

    r.to_torch("cpu")

    for step_key in [0, 1]:
        for value in r.observations[step_key].values():
            assert isinstance(value, torch.Tensor)
        for value in r.actions[step_key].values():
            assert isinstance(value, torch.Tensor)
