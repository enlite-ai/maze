"""Test the no dict action wrapper"""
import numpy as np
from gym import spaces

from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.no_dict_spaces_wrapper import NoDictSpacesWrapper


def test_no_dict_action_wrapper():
    """ gym env wrapper unit test """
    base_env = GymMazeEnv(env="CartPole-v0")
    env = NoDictSpacesWrapper.wrap(base_env)

    # test action
    assert isinstance(env.action_space, spaces.Discrete)
    assert isinstance(env.action_spaces_dict, dict)

    action = env.action_space.sample()
    out_action = env.action(action)
    assert isinstance(out_action, dict)
    assert out_action['action'] == action

    assert env.action_space.contains(env.reverse_action(out_action))
    assert env.reverse_action(out_action) == action

    # test observation
    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(env.observation_spaces_dict, dict)

    assert isinstance(env.observation_space.sample(), np.ndarray)
    assert env.observation_space.contains(env.observation_space.sample())
    assert env.observation_space.contains(env.reset())
