import os
import pickle

import numpy as np

from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord
from maze.core.wrappers.spaces_recording_wrapper import SpacesRecordingWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


def test_records_episode_with_correct_data():
    env = build_dummy_maze_env()
    env = SpacesRecordingWrapper.wrap(env, dump_file_prefix="test_record_")

    actions = []
    observations = []

    observation = env.reset()
    for i in range(5):
        observations.append(observation)
        action = env.action_space.sample()
        actions.append(action)
        observation, _, _, _ = env.step(action)

    episode_id = env.get_episode_id()
    expected_file_path = "test_record_" + str(episode_id) + ".pkl"
    assert not expected_file_path in os.listdir(".")

    # Now dump and load the data
    env.reset()
    assert expected_file_path in os.listdir(".")
    with open(expected_file_path, "rb") as in_f:
        episode_record = pickle.load(in_f)

    # Check the contents
    assert isinstance(episode_record, SpacesTrajectoryRecord)
    assert len(episode_record.step_records) == len(actions)
    for record, observation, action in zip(episode_record.step_records, observations, actions):
        assert len(record.actions) == 1 and len(record.observations) == 1  # single-step env
        for obs_key in observation:
            assert np.allclose(record.observations[0][obs_key], observation[obs_key])
        for act_key in action:
            assert np.allclose(record.actions[0][act_key], action[act_key])

