"""
This script includes unit tests for the dummy environments
"""
from gym import spaces

from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env, \
    build_dummy_maze_env_with_structured_core_env


def test_dummy_maze_env():
    """
    Unit test for the DummyEnvironment
    """
    env = build_dummy_maze_env()
    _ = env.reset()

    for _ in range(10):
        action = env.action_space.sample()
        observation, _, _, _ = env.step(action)

    # check if actor is done
    assert not env.is_actor_done()

    env.close()


def test_dummy_struct_env():
    """
    Unit test for the DummyStructuredEnvironment
    """
    maze_env = build_dummy_maze_env()
    env = DummyStructuredEnvironment(maze_env)
    env.reset()

    # check observation space
    assert isinstance(env.observation_space, spaces.Dict)

    for i in range(10):
        action = env.action_spaces_dict[env.actor_id()[0]].sample()
        observation, _, _, _ = env.step(action)

    env.close()


def test_dummy_struct_core_env():
    env = build_dummy_maze_env_with_structured_core_env()
    env.reset()

    for i in range(10):
        assert env.actor_id()[1] == i % 2
        action = env.action_space.sample()
        observation, _, _, _ = env.step(action)

    env.close()
