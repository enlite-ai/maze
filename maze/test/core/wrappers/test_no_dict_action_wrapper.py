"""Test the no dict action wrapper"""
from gym import spaces

from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.no_dict_action_wrapper import NoDictActionWrapper


def test_no_dict_action_wrapper():
    """ gym env wrapper unit test """
    base_env = GymMazeEnv(env="CartPole-v0")
    env = NoDictActionWrapper.wrap(base_env)

    assert isinstance(env.action_space, spaces.Discrete)
    assert isinstance(env.action_spaces_dict, dict)

    action = env.action_space.sample()
    out_action = env.action(action)
    assert isinstance(out_action, dict)
    assert out_action['action'] == action

    assert env.action_space.contains(env.reverse_action(out_action))
    assert env.reverse_action(out_action) == action

