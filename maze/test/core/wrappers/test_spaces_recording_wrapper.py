import os
import pickle

import numpy as np

from maze.core.trajectory_recording.datasets.in_memory_dataset import InMemoryDataset
from maze.core.trajectory_recording.datasets.trajectory_processor import IdentityTrajectoryProcessor
from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord
from maze.core.wrappers.spaces_recording_wrapper import SpacesRecordingWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env, build_dummy_structured_env


def test_records_episode_with_correct_data():
    env = build_dummy_maze_env()
    env = SpacesRecordingWrapper.wrap(env, output_dir="space_records")

    actions = []
    observations = []

    observation = env.reset()
    for _ in range(5):
        observations.append(observation)
        action = env.action_space.sample()
        actions.append(action)
        observation, _, _, _ = env.step(action)

    episode_id = env.get_episode_id()
    expected_file_path = str(episode_id) + ".pkl"
    assert not os.path.exists("space_records")

    # Now dump and load the data
    env.reset()
    assert expected_file_path in os.listdir("space_records")
    with open("space_records/" + expected_file_path, "rb") as in_f:
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


def test_records_multiple_episodes():
    env = build_dummy_maze_env()
    env = SpacesRecordingWrapper.wrap(env, output_dir="space_records")

    env.reset()
    for _ in range(5):
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)
        env.reset()

    dumped_files = os.listdir("space_records")
    assert len(dumped_files) == 5

    for file_path in dumped_files:
        with open("space_records/" + file_path, "rb") as in_f:
            episode_record = pickle.load(in_f)

        assert isinstance(episode_record, SpacesTrajectoryRecord)
        assert len(episode_record.step_records) == 10


def test_handles_multi_step_scenarios():
    env = build_dummy_structured_env()
    env = SpacesRecordingWrapper.wrap(env, output_dir="space_records")

    env.reset()
    for _ in range(6):
        action = env.action_space.sample()
        env.step(action)

    assert len(env.episode_record.step_records) == 3
    for step_record in env.episode_record.step_records:
        assert len(step_record.actions) == 2
        assert len(step_record.observations) == 2


def test_compatibility_with_dataset():
    env = build_dummy_maze_env()
    env = SpacesRecordingWrapper.wrap(env, output_dir="space_records")

    # Generate 5 episodes, 10 steps each
    env.reset()
    for _ in range(5):
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)
        env.reset()

    dataset = InMemoryDataset(
        n_workers=2,
        conversion_env_factory=None,
        input_data="space_records",
        trajectory_processor=IdentityTrajectoryProcessor(),
        deserialize_in_main_thread=False
    )

    assert len(dataset) == 5 * 10
