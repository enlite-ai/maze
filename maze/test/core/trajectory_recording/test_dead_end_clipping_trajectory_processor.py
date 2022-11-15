"""Test in memory data set with the dead end clipping processing method"""
import os

from maze.core.trajectory_recording.datasets.in_memory_dataset import InMemoryDataset
from maze.core.trajectory_recording.datasets.trajectory_processor import DeadEndClippingTrajectoryProcessor, \
    IdentityTrajectoryProcessor
from maze.core.utils.factory import Factory
from maze.core.wrappers.maze_gym_env_wrapper import make_gym_maze_env
from maze.test.shared_test_utils.run_maze_utils import run_maze_job


def test_sequential_data_load_from_directory():
    """Test loading trajectories of multiple episodes in parallel into an in-memory dataset. (Each
    data-loader process reads the files assigned to it.)"""
    # Heuristics rollout
    rollout_config = {
        "configuration": "test",
        "env": "gym_env",
        "env.name": "CartPole-v0",
        "policy": "random_policy",
        "runner": "sequential",
        "runner.n_episodes": 1,
        "runner.record_trajectory": True,
        "runner.max_episode_steps": 18,
        "seeding.env_base_seed": 12345,
        "seeding.agent_base_seed": 12345,
    }
    run_maze_job(rollout_config, config_module="maze.conf", config_name="conf_rollout")

    dataset = InMemoryDataset(
        n_workers=1,
        conversion_env_factory=lambda: make_gym_maze_env("CartPole-v0"),
        input_data="trajectory_data",
        trajectory_processor=DeadEndClippingTrajectoryProcessor(clip_k=2),
        deserialize_in_main_thread=False,
    )


def test_sequential_data_load_from_directory_clipped():
    """Test loading trajectories of multiple episodes in parallel into an in-memory dataset. (Each
    data-loader process reads the files assigned to it.)"""
    # Heuristics rollout
    rollout_config = {
        "configuration": "test",
        "env": "gym_env",
        "env.name": "CartPole-v0",
        "policy": "random_policy",
        "runner": "sequential",
        "runner.n_episodes": 1,
        "runner.record_trajectory": True,
        "runner.max_episode_steps": 20,
        "seeding.env_base_seed": 12345,
        "seeding.agent_base_seed": 12345,
    }
    run_maze_job(rollout_config, config_module="maze.conf", config_name="conf_rollout")

    dataset = InMemoryDataset(
        n_workers=1,
        conversion_env_factory=lambda: make_gym_maze_env("CartPole-v0"),
        input_data="trajectory_data",
        trajectory_processor=DeadEndClippingTrajectoryProcessor(clip_k=2),
        deserialize_in_main_thread=False
    )


def test_parallel_data_load_from_directory_clipped(tmpdir):
    """Test loading trajectories of multiple episodes in parallel into an in-memory dataset. (Each
    data-loader process reads the files assigned to it.)"""
    os.chdir(tmpdir)

    # Heuristics rollout
    rollout_config = {
        "configuration": "test",
        "env": "gym_env",
        "env.name": "CartPole-v0",
        "policy": "random_policy",
        "runner": "sequential",
        "runner.n_episodes": 2,
        "runner.record_trajectory": True,
        "runner.max_episode_steps": 50,
        "seeding.env_base_seed": 12345,
        "seeding.agent_base_seed": 12345,
    }
    run_maze_job(rollout_config, config_module="maze.conf", config_name="conf_rollout")

    dataset = InMemoryDataset(
        n_workers=2,
        conversion_env_factory=lambda: make_gym_maze_env("CartPole-v0"),
        input_data="trajectory_data",
        trajectory_processor=IdentityTrajectoryProcessor(),
        deserialize_in_main_thread=False
    )

    dataset_clipped = InMemoryDataset(
        n_workers=2,
        conversion_env_factory=lambda: make_gym_maze_env("CartPole-v0"),
        input_data="trajectory_data",
        trajectory_processor=DeadEndClippingTrajectoryProcessor(clip_k=2),
        deserialize_in_main_thread=False
    )

    assert len(dataset) - 4 == len(dataset_clipped)


def test_parallel_data_load_from_directory_clipped_from_hydra(tmpdir):
    """Test loading trajectories of multiple episodes in parallel into an in-memory dataset. (Each
    data-loader process reads the files assigned to it.)"""
    os.chdir(tmpdir)

    # Heuristics rollout
    rollout_config = {
        "configuration": "test",
        "env": "gym_env",
        "env.name": "CartPole-v0",
        "policy": "random_policy",
        "runner": "sequential",
        "runner.n_episodes": 2,
        "runner.record_trajectory": True,
        "runner.max_episode_steps": 50,
        "seeding.env_base_seed": 12345,
        "seeding.agent_base_seed": 12345,
    }
    run_maze_job(rollout_config, config_module="maze.conf", config_name="conf_rollout")

    hydra_config = {
        '_target_': 'maze.core.trajectory_recording.datasets.in_memory_dataset.InMemoryDataset',
        'n_workers': 2,
        'conversion_env_factory': lambda: make_gym_maze_env("CartPole-v0"),
        'input_data': 'trajectory_data',
        'deserialize_in_main_thread': False,
        'trajectory_processor': {
            '_target_': 'maze.core.trajectory_recording.datasets.trajectory_processor.DeadEndClippingTrajectoryProcessor',
            'clip_k': 2
        }
    }

    dataset_reference = InMemoryDataset(
        n_workers=2,
        conversion_env_factory=lambda: make_gym_maze_env("CartPole-v0"),
        input_data="trajectory_data",
        trajectory_processor=IdentityTrajectoryProcessor(),
        deserialize_in_main_thread=False
    )

    dataset_clipped = Factory(InMemoryDataset).instantiate(hydra_config)

    assert len(dataset_reference) - 4 == len(dataset_clipped)
