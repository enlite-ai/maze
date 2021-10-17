"""Test the Discretize actions wrapper"""
from typing import Dict

import numpy as np
import pytest
from gym import spaces

from maze.core.wrappers.discretize_actions_wrapper import DiscretizeActionsWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.split_actions_wrapper import SplitActionsWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


def test_discrete_action_wrapper() -> None:
    """ DiscretizeActions wrapper unit test """
    base_env = GymMazeEnv(env="LunarLanderContinuous-v2")
    split_config = {
        'action': {
            'action_up': {
                'indices': [0]
            },
            'action_side': {
                'indices': [1]
            }
        }
    }
    base_env = SplitActionsWrapper.wrap(base_env, split_config=split_config)
    discretization_config = {
        'action_up': {
            'num_bins': 4,
            'low': -1,
            'high': 1
        }
    }
    env = DiscretizeActionsWrapper.wrap(base_env, discretization_config=discretization_config)

    for _, action_space in env.action_spaces_dict.items():
        assert isinstance(action_space, spaces.Dict)
    for action_name, new_action_config in discretization_config.items():
        assert action_name in env.action_space.spaces.keys()
        assert isinstance(env.action_space[action_name], spaces.Discrete)
        assert env.action_space[action_name].n == new_action_config['num_bins']

    action = env.action_space.sample()
    reverse_action = env.action(action)
    restored_action = env.reverse_action(reverse_action)
    assert isinstance(action, dict)
    for action_name, action_config in discretization_config.items():
        assert action_name in action
        assert action[action_name] in spaces.Discrete(action_config['num_bins'])
        assert reverse_action[action_name] in base_env.action_space[action_name]
        assert np.all(restored_action[action_name] == action[action_name])


def _test_dummy_env_for_discretization_config(discretization_config: Dict[str, Dict]) -> None:
    """Test the Discretize action wrapper on the dummy env for a given discretization_config.

    :param discretization_config: The Discretize actions config to apply and test.
    """
    base_env = build_dummy_maze_env()

    env = DiscretizeActionsWrapper.wrap(base_env, discretization_config=discretization_config)

    assert isinstance(env.action_space, spaces.Dict)
    for action_name, new_action_config in discretization_config.items():
        assert action_name in env.action_space.spaces.keys()
        if base_env.action_space[action_name].shape[-1] == 1:
            assert isinstance(env.action_space[action_name], spaces.Discrete)
            assert env.action_space[action_name].n == new_action_config['num_bins']
        else:
            assert isinstance(env.action_space[action_name], spaces.MultiDiscrete)
            assert np.all(env.action_space[action_name].nvec == \
                          np.array([new_action_config['num_bins']] * base_env.action_space[action_name].shape[-1]))

    for i in range(1):
        action = env.action_space.sample()
        env.step(action)
        reverse_action = env.action(action)
        restored_action = env.reverse_action(reverse_action)
        assert isinstance(action, dict)
        for action_name, action_config in discretization_config.items():
            assert action_name in action
            if base_env.action_space[action_name].shape[-1] == 1:
                assert action[action_name] in spaces.Discrete(action_config['num_bins'])
            else:
                nvec = np.array([action_config['num_bins']] * base_env.action_space[action_name].shape[-1])
                assert action[action_name] in spaces.MultiDiscrete(nvec)
            assert reverse_action[action_name] in base_env.action_space[action_name]
            assert np.all(restored_action[action_name] == action[action_name])


def test_dummy_env_discretize_actions_continuous() -> None:
    """Test for continuous actions"""

    discretization_config = {
        'action_0_2': {
            'num_bins': 10
        },
        'action_2_0': {
            'num_bins': 5,
            'low': [-5, 0, -1, -0, -5],
            'high': 5
        }
    }

    _test_dummy_env_for_discretization_config(discretization_config)


def test_dummy_env_discretize_actions_continuous_2() -> None:
    """Test for continuous actions"""

    discretization_config = {
        'action_0_2': {
            'num_bins': 10,
            'high': [1, 1, 1, 1, 1],
            'low': [-1, 0, -1, 0, -1]
        },
        'action_2_0': {
            'num_bins': 5,
            'high': [5, 0, 1, 0, 5],
            'low': -1
        }
    }

    _test_dummy_env_for_discretization_config(discretization_config)


def test_dummy_env_discretize_actions_continuous_wrong() -> None:
    """Test for continuous actions"""

    discretization_config = {
        'action_0_2': {
            'num_bins': 10
        },
        'action_2_0': {
            'num_bins': 5,
            'high': [5, -2, 1, 0, 5],
            'low': -1
        }
    }

    with pytest.raises(AssertionError):
        _test_dummy_env_for_discretization_config(discretization_config)
