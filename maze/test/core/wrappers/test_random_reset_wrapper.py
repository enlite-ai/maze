""" Contains tests for the random reset wrapper. """
from maze.core.wrappers.random_reset_wrapper import RandomResetWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


def test_random_reset_wrapper():
    """ random reset wrapper unit tests """
    env = build_dummy_maze_env()
    env = RandomResetWrapper.wrap(env, min_skip_steps=2, max_skip_steps=3)
    env.seed(1234)
    env.reset()
