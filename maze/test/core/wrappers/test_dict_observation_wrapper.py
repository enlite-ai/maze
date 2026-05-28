"""Tests for DictObservationWrapper."""

from __future__ import annotations

from unittest.mock import MagicMock

from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.wrappers.dict_observation_wrapper import DictObservationWrapper

import numpy as np
from gymnasium import spaces


def _make_mock_box_env(obs_space: spaces.Box) -> SimulatedEnvMixin:
    """Create a minimal mock environment with the given Box observation space.

    :param obs_space: The Box observation space.
    :return: A mock environment with the given observation space.
    """
    env = MagicMock(spec=SimulatedEnvMixin)
    env.observation_space = obs_space
    env.observation_spaces_dict = {0: obs_space}
    return env


def test_dict_obs_space_is_dict():
    """Output observation space must be a Dict containing the original space under the 'observation' key."""
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    env = DictObservationWrapper(_make_mock_box_env(obs_space))

    assert isinstance(env.observation_space, spaces.Dict)
    assert 'observation' in env.observation_space.spaces
    assert env.observation_space.spaces['observation'] == obs_space


def test_dict_observation_spaces_dict_keys_are_preserved():
    """observation_spaces_dict keys must match the underlying env (DictObservationWrapper does not remap them)."""
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    env = DictObservationWrapper(_make_mock_box_env(obs_space))

    assert set(env.observation_spaces_dict.keys()) == {0}


def test_dict_observation_wrapper_wraps_obs_in_dict():
    """observation() must wrap the array in a single-key dict and the result must be contained in the space."""
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    env = DictObservationWrapper(_make_mock_box_env(obs_space))

    sample = obs_space.sample()
    result = env.observation(sample)

    assert isinstance(result, dict)
    assert 'observation' in result
    assert env.observation_space.contains(result)


def test_dict_observation_wrapper_casts_to_float32():
    """observation() must cast to float32 regardless of the input dtype."""
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64)
    env = DictObservationWrapper(_make_mock_box_env(obs_space))

    sample = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    result = env.observation(sample)

    assert result['observation'].dtype == np.float32


def test_dict_observation_wrapper_none_passthrough():
    """None observations must be returned as None without raising."""
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    env = DictObservationWrapper(_make_mock_box_env(obs_space))

    assert env.observation(None) is None
