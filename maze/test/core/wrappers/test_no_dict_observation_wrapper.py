"""Test the no dict observation wrapper"""

from __future__ import annotations

from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.no_dict_observation_wrapper import NoDictObservationWrapper

import numpy as np
from gymnasium import spaces


def test_no_dict_action_wrapper():
    """gym env wrapper unit test"""
    base_env = GymMazeEnv(env='CartPole-v1', render_mode=None)
    env = NoDictObservationWrapper.wrap(base_env)

    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(env.observation_spaces_dict, dict)

    assert isinstance(env.observation_space.sample(), np.ndarray)
    assert env.observation_space.contains(env.observation_space.sample())
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)


def test_no_dict_observation_wrapper_none_passthrough():
    """None observations must be returned as None without raising."""
    base_env = GymMazeEnv(env='CartPole-v1', render_mode=None)
    env = NoDictObservationWrapper.wrap(base_env)
    assert env.observation(None) is None
