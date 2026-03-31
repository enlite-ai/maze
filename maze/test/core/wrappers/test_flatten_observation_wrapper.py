"""Tests for FlattenDictObservationWrapper"""

from __future__ import annotations

from unittest.mock import MagicMock

from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.wrappers.flatten_observation_wrapper import FlattenDictObservationWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.spaces import flatten_space
from gymnasium.spaces import utils as space_utils


def _expected_flat_dim(obs_space: spaces.Space) -> int:
    """
    Returns the expected flattened dimension of the observation space

    :param obs_space: The observation space
    :return: The expected flattened dimension of the observation space
    """
    return space_utils.flatdim(obs_space)


# CartPole: Box obs space (trivial case — should pass through unchanged in shape)


def test_flat_obs_space_is_box():
    """Output observation space must always be a flat Box."""
    base_env = GymMazeEnv(env='CartPole-v1', render_mode=None)
    env = FlattenDictObservationWrapper(base_env)

    assert isinstance(env.observation_space, spaces.Box)

    expected_dim = _expected_flat_dim(base_env.observation_space)
    assert env.observation_space.shape == (expected_dim,)
    assert env.observation_space.dtype == np.float32

    obs, _ = env.reset()
    assert env.observation_space.contains(obs), f'reset obs {obs} not contained in {env.observation_space}'

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# Dict observation space
# ---------------------------------------------------------------------------


def _make_mock_dict_env(obs_space: spaces.Space) -> SimulatedEnvMixin:
    """
    Create a minimal mock environment with the given observation space.

    :param obs_space: The observation space
    :return: A mock environment with the given observation space
    """

    env = MagicMock(spec=SimulatedEnvMixin)
    env.observation_space = obs_space
    env.observation_spaces_dict = {0: obs_space}
    return env


def test_dict_obs_flat_dim():
    """Flattened dim must equal sum of individual sub-space flatdims."""
    obs_space = spaces.Dict(
        {
            'z_last': spaces.Box(low=99.0, high=100.0, shape=(1,), dtype=np.float32),
            'a_first': spaces.Box(low=-100.0, high=-99.0, shape=(1,), dtype=np.float32),
        }
    )
    base_env = FlattenDictObservationWrapper(_make_mock_dict_env(obs_space))
    env = FlattenDictObservationWrapper(base_env)

    expected_dim = _expected_flat_dim(base_env.observation_space)  # 2 + 3 + 4 = 9
    assert env.observation_space.shape == (expected_dim,), (
        f'Expected shape ({expected_dim},), got {env.observation_space.shape}'
    )

    expected_space = flatten_space(base_env.observation_space)

    np.testing.assert_array_equal(env.observation_space.low, expected_space.low)
    np.testing.assert_array_equal(env.observation_space.high, expected_space.high)

    # Test if the sample from the flattened observation space is 1D
    sample = base_env.observation_space.sample()
    flat = env.observation(sample)
    assert flat.ndim == 1
    assert flat.shape == env.observation_space.shape

    # assert that tipo is float32
    assert flat.dtype == np.float32

    # assert that keys are the same
    assert set(env.observation_spaces_dict.keys()) == set(base_env.observation_spaces_dict.keys())


def test_observation_spaces_dict_values_are_flat_boxes():
    """
    Test if the observation spaces in the observation_spaces_dict are flat Boxes.
    """
    obs_space = spaces.Dict(
        {
            'z_last': spaces.Box(low=99.0, high=100.0, shape=(1,), dtype=np.float32),
            'a_first': spaces.Box(low=-100.0, high=-99.0, shape=(1,), dtype=np.float32),
        }
    )
    base_env = FlattenDictObservationWrapper(_make_mock_dict_env(obs_space))
    env = FlattenDictObservationWrapper(base_env)

    for k, sp in env.observation_spaces_dict.items():
        assert isinstance(sp, spaces.Box), f'Key {k!r} has space {type(sp)}, expected Box'
        assert sp.shape[0] == _expected_flat_dim(base_env.observation_spaces_dict[k]), f'Flatdim mismatch for key {k!r}'


# Determinism: sorted keys guarantee stable flattening order


def test_flattening_is_deterministic_across_calls():
    """Two calls to observation() on the same input must yield bit-identical arrays."""
    obs_space = spaces.Dict(
        {
            'z_last': spaces.Box(low=99.0, high=100.0, shape=(1,), dtype=np.float32),
            'a_first': spaces.Box(low=-100.0, high=-99.0, shape=(1,), dtype=np.float32),
        }
    )
    base_env = FlattenDictObservationWrapper(_make_mock_dict_env(obs_space))
    env = FlattenDictObservationWrapper(base_env)

    np.random.default_rng(0)
    sample = base_env.observation_space.sample()

    flat_a = env.observation(sample)
    flat_b = env.observation(sample)

    np.testing.assert_array_equal(flat_a, flat_b)


def test_sorted_keys_determine_flat_order():
    """Values from alphabetically earlier keys must appear first in the flat array."""
    obs_space = spaces.Dict(
        {
            'z_last': spaces.Box(low=99.0, high=100.0, shape=(1,), dtype=np.float32),
            'a_first': spaces.Box(low=-100.0, high=-99.0, shape=(1,), dtype=np.float32),
        }
    )
    env = FlattenDictObservationWrapper(_make_mock_dict_env(obs_space))

    obs = {'a_first': np.array([-99.5], dtype=np.float32), 'z_last': np.array([99.5], dtype=np.float32)}

    flat = env.observation(obs)

    assert flat[0] == pytest.approx(-99.5)
    assert flat[1] == pytest.approx(99.5)

    obs = {
        'z_last': np.array([99.5], dtype=np.float32),
        'a_first': np.array([-99.5], dtype=np.float32),
    }

    flat = env.observation(obs)

    assert flat[0] == pytest.approx(-99.5)
    assert flat[1] == pytest.approx(99.5)


def test_nested_dict_obs_flattening_correctness():
    """
    Verifies correct flattening of a deeply nested mixed observation space.
    Total expected flat dim: 4 + 2 + 3 + 3 + 1 = 13
    """

    obs_space = spaces.Dict(
        {
            'timestep': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'sensor': spaces.Dict(
                {
                    'reading': spaces.MultiBinary(3),
                    'depth': spaces.Box(low=0.0, high=10.0, shape=(3,), dtype=np.float32),
                }
            ),
            'agent': spaces.Dict(
                {
                    'position': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                    'orientation': spaces.Discrete(4),
                }
            ),
        }
    )

    env = FlattenDictObservationWrapper(_make_mock_dict_env(obs_space))

    obs = {
        'timestep': np.array([0.5], dtype=np.float32),
        'sensor': {
            'reading': np.array([1, 0, 1], dtype=np.int8),
            'depth': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        },
        'agent': {
            'position': np.array([0.3, -0.7], dtype=np.float32),
            'orientation': 2,  # one-hot -> [0, 0, 1, 0]
        },
    }

    flat = env.observation(obs)

    assert flat.ndim == 1
    assert flat.dtype == np.float32
    assert flat.shape == (13,), f'Expected (13,), got {flat.shape}'
    assert env.observation_space.contains(flat)

    # --- sorted key order: agent < sensor < timestep ---
    # agent.orientation: Discrete(4) one-hot for value 2
    np.testing.assert_array_equal(flat[0:4], [0.0, 0.0, 1.0, 0.0])

    # agent.position
    np.testing.assert_array_almost_equal(flat[4:6], [0.3, -0.7])
    # sensor.depth (depth < reading alphabetically)
    np.testing.assert_array_almost_equal(flat[6:9], [1.0, 2.0, 3.0])
    # sensor.reading
    np.testing.assert_array_equal(flat[9:12], [1.0, 0.0, 1.0])
    # timestep
    np.testing.assert_array_almost_equal(flat[12:13], [0.5])
