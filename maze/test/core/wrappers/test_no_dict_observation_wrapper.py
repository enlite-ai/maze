"""Test the no dict observation wrapper"""
import numpy as np
from gym import spaces
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.no_dict_observation_wrapper import NoDictObservationWrapper


def test_no_dict_action_wrapper():
    """ gym env wrapper unit test """
    base_env = GymMazeEnv(env="CartPole-v0")
    env = NoDictObservationWrapper.wrap(base_env)

    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(env.observation_spaces_dict, dict)

    assert isinstance(env.observation_space.sample(), np.ndarray)
    assert env.observation_space.contains(env.observation_space.sample())
    assert env.observation_space.contains(env.reset())

