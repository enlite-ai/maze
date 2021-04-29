"""Contains tests for the export gif wrapper."""
import glob

from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.export_gif_wrapper import ExportGifWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


def assert_gif_export(env: MazeEnv) -> None:
    """Checks if gif got exported correctly."""
    env.reset()
    for _ in range(3):
        env.step(env.action_space.sample())
    env.close()

    # check if gif was exported
    gif_files = glob.glob("*.gif")
    assert len(gif_files) == 1


def test_gym_env_gif_export():
    """ Gif export unit test """
    env = GymMazeEnv(env="CartPole-v0")
    env = ExportGifWrapper.wrap(env, export=True, duration=0.1)
    assert_gif_export(env)


def test_maze_env_gif_export():
    """ Gif export unit test """
    env = build_dummy_maze_env()
    env = ExportGifWrapper.wrap(env, export=True, duration=0.1)
    assert_gif_export(env)
