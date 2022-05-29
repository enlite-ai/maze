""" Contains action recording wrapper unit tests. """
import os
import pickle

import numpy as np
import pytest

from maze.core.env.structured_env import ActorID
from maze.core.trajectory_recording.records.action_record import ActionRecord
from maze.core.wrappers.action_recording_wrapper import ActionRecordingWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env, build_dummy_structured_env


def test_enforce_episode_seeding():
    with pytest.raises(AssertionError):
        env = build_dummy_maze_env()
        env = ActionRecordingWrapper.wrap(env, record_maze_actions=True, record_actions=True,
                                          output_dir="action_records")
        env.reset()


def test_records_episode_with_correct_data():
    env = build_dummy_maze_env()
    env = ActionRecordingWrapper.wrap(env, record_maze_actions=True, record_actions=True,
                                      output_dir="action_records")

    actions = []

    env.seed(1234)
    env.reset()
    cum_reward = 0.0
    for _ in range(5):
        action = env.action_space.sample()
        actions.append(action)
        observation, rew, _, _ = env.step(action)
        cum_reward += rew

    episode_id = env.get_episode_id()
    expected_file_path = str(episode_id) + ".pkl"
    assert not expected_file_path in os.listdir("action_records")

    # Now dump and load the data
    env.seed(1234)
    env.reset()
    assert expected_file_path in os.listdir("action_records")
    with open("action_records/" + expected_file_path, "rb") as in_f:
        action_record = pickle.load(in_f)

    assert action_record.cum_action_record_reward == cum_reward

    # Check the contents
    assert isinstance(action_record, ActionRecord)
    assert len(action_record) == len(actions)
    for i in range(5):
        recorded_action = action_record.get_agent_action(i, ActorID(0, 0))
        for k in recorded_action.keys():
            assert np.all(recorded_action[k] == actions[i][k])


def test_records_multiple_episodes():
    env = build_dummy_maze_env()
    env = ActionRecordingWrapper.wrap(env, record_maze_actions=True, record_actions=True,
                                      output_dir="action_records")

    env.seed(1234)
    env.reset()
    for _ in range(5):
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)
        env.seed(1234)
        env.reset()

    dumped_files = os.listdir("action_records")
    assert len(dumped_files) == 5

    for file_path in dumped_files:
        with open("action_records/" + file_path, "rb") as in_f:
            action_record = pickle.load(in_f)

        assert isinstance(action_record, ActionRecord)
        assert len(action_record) == 10


def test_handles_multi_step_scenarios():
    env = build_dummy_structured_env()
    env = ActionRecordingWrapper.wrap(env, record_maze_actions=False, record_actions=True,
                                      output_dir="action_records")

    env.seed(1234)
    env.reset()
    for _ in range(6):
        action = env.action_space.sample()
        env.step(action)

    assert len(env.action_record) == 3
    for i in range(3):
        assert ActorID(0, 0) in env.action_record.agent_actions[i]
        assert ActorID(1, 0) in env.action_record.agent_actions[i]
