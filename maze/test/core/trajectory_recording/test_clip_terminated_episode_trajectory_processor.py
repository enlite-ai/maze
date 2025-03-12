"""Test the clip terminated episode trajectory processor."""
import copy

from maze.core.trajectory_recording.datasets.in_memory_dataset import InMemoryDataset
from maze.core.trajectory_recording.datasets.trajectory_processor import ClipTerminatedEpisodeTrajectoryProcessor
from maze.core.trajectory_recording.datasets.utils import retrieve_done_info
from maze.test.shared_test_utils.run_maze_utils import run_maze_job


def test_terminated_trajectory():
    """Test loading trajectories of multiple episodes in parallel into an in-memory dataset. (Each
    data-loader process reads the files assigned to it.)"""
    # Heuristics rollout
    rollout_config = {
        "configuration": "test",
        "env": "gym_env",
        "env.name": "CartPole-v1",
        "policy": "random_policy",
        "runner": "sequential",
        "runner.n_episodes": 1,
        "runner.record_trajectory": True,
        "runner.max_episode_steps": 100,
        "seeding.env_base_seed": 12345,
        "seeding.agent_base_seed": 12345,
    }
    run_maze_job(rollout_config, config_module="maze.conf", config_name="conf_rollout")

    trajectory_files = InMemoryDataset._read_input_data_to_list('trajectory_data')
    test_file = trajectory_files[0]
    trajectory_record = list(InMemoryDataset.deserialize_trajectory(test_file))[0]

    terminated, truncated, info = retrieve_done_info(trajectory_record)

    print(len(trajectory_record))
    print(info)
    assert terminated
    assert not truncated

    processed_trajectory = ClipTerminatedEpisodeTrajectoryProcessor(clip_k=2).pre_process(
        copy.deepcopy(trajectory_record))
    assert len(processed_trajectory) == len(trajectory_record) - 2

    processed_trajectory = ClipTerminatedEpisodeTrajectoryProcessor(clip_k=5).pre_process(
        copy.deepcopy(trajectory_record))
    assert len(processed_trajectory) == len(trajectory_record) - 5

    processed_trajectory = ClipTerminatedEpisodeTrajectoryProcessor(clip_k=0).pre_process(
        copy.deepcopy(trajectory_record))
    assert len(processed_trajectory) == len(trajectory_record) - 0

    processed_trajectory = ClipTerminatedEpisodeTrajectoryProcessor(clip_k=1).pre_process(
        copy.deepcopy(trajectory_record))
    assert len(processed_trajectory) == len(trajectory_record) - 1

    processed_trajectory = ClipTerminatedEpisodeTrajectoryProcessor(clip_k=100).pre_process(
        copy.deepcopy(trajectory_record))
    assert len(processed_trajectory) == 0


def test_clip_terminated_false():
    """Test loading trajectories of multiple episodes in parallel into an in-memory dataset. (Each
    data-loader process reads the files assigned to it.)"""
    # Heuristics rollout
    rollout_config = {
        "configuration": "test",
        "env": "gym_env",
        "env.name": "CartPole-v1",
        "policy": "random_policy",
        "runner": "sequential",
        "runner.n_episodes": 1,
        "runner.record_trajectory": True,
        "runner.max_episode_steps": 9,
        "seeding.env_base_seed": 12345,
        "seeding.agent_base_seed": 12345,
    }
    run_maze_job(rollout_config, config_module="maze.conf", config_name="conf_rollout")

    trajectory_files = InMemoryDataset._read_input_data_to_list('trajectory_data')
    test_file = trajectory_files[0]
    trajectory_record = list(InMemoryDataset.deserialize_trajectory(test_file))[0]

    terminated, truncated, info = retrieve_done_info(trajectory_record)

    assert truncated
    assert not terminated

    processed_trajectory = ClipTerminatedEpisodeTrajectoryProcessor(clip_k=2).pre_process(trajectory_record)
    assert len(processed_trajectory) == len(trajectory_record)
