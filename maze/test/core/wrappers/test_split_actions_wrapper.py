"""Test the split action wrapper"""
from typing import Dict

import numpy as np
import pytest
from gym import spaces

from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.split_actions_wrapper import SplitActionsWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


def test_split_action_wrapper() -> None:
    """ gym env wrapper unit test """
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
    env = SplitActionsWrapper.wrap(base_env, split_config=split_config)
    assert isinstance(env.action_space, spaces.Dict)
    for action_name, new_action_config in split_config['action'].items():
        assert action_name in env.action_space.spaces.keys()
        assert isinstance(env.action_space[action_name], spaces.Box)
        assert env.action_space[action_name].shape[-1] == len(['indices'])

    action = env.action_space.sample()
    assert isinstance(action, dict)
    for action_name in split_config['action'].keys():
        assert action_name in action

    base_env_action = base_env.action_space.sample()
    split_action = env.reverse_action(base_env_action)

    assert isinstance(split_action, dict)
    for action_name, new_action_config in split_config['action'].items():
        assert action_name in split_action
        assert np.stack([base_env_action['action'][select_index] for select_index in new_action_config['indices']]) \
               == split_action[action_name]

    assert np.all(base_env_action['action'] == env.action(split_action)['action'])


def _test_dummy_env_for_split_config(split_config: Dict[str, Dict]) -> None:
    """Test the split action wrapper on the dummy env for a given split_config

    :param split_config: The split action config to apply and test.
    """
    base_env = build_dummy_maze_env()

    env = SplitActionsWrapper.wrap(base_env, split_config=split_config)

    _ = env.action_space

    for key, sub_actions in split_config.items():
        for sub_key in sub_actions.keys():
            assert sub_key in env.action_space.spaces.keys()
        assert key not in env.action_space.spaces.keys()

    assert env.action_spaces_dict[0] == env.action_space

    base_env_action = base_env.action_space.sample()
    split_action = env.reverse_action(base_env_action)

    for key, sub_actions in split_config.items():
        for sub_key in sub_actions.keys():
            assert sub_key in split_action
        assert key not in split_action

    for key in base_env.action_space.spaces.keys():
        assert np.all(base_env_action[key] == env.action(split_action)[key])


def test_wrong_split_action_space_config() -> None:
    """Test a few not allowed configs"""
    base_env = build_dummy_maze_env()

    with pytest.raises(AssertionError):
        split_config = {
            'action_0_2': {
                'action_up': {
                    'indices': [0, 2, 5]
                },
                'action_side': {
                    'indices': [1, 3, 4]
                }
            }
        }
        _ = SplitActionsWrapper.wrap(base_env, split_config=split_config)

    with pytest.raises(AssertionError):
        split_config = {
            'action_0_2': {
                'action_up': {
                    'indices': [2]
                },
                'action_side': {
                    'indices': [1, 3, 4]
                }
            }
        }
        _ = SplitActionsWrapper.wrap(base_env, split_config=split_config)

    with pytest.raises(AssertionError):
        split_config = {
            'action_0_2': {
                'action_up': {
                    'indices': [2]
                },
                'action_side': {
                    'indices': [1, 3, 4]
                }
            }
        }
        _ = SplitActionsWrapper.wrap(base_env, split_config=split_config)

    with pytest.raises(AssertionError):
        split_config = {
            'action_0_2': {
                'action_up': {
                    'indices': [2, 2]
                },
                'action_side': {
                    'indices': [1, 3, 4]
                }
            }
        }
        _ = SplitActionsWrapper.wrap(base_env, split_config=split_config)

    with pytest.raises(NotImplementedError):
        split_config = {
            'action_0_0': {
                'action_up': {
                    'indices': [2, 2]
                },
                'action_side': {
                    'indices': [1, 3, 4]
                }
            }
        }
        _ = SplitActionsWrapper.wrap(base_env, split_config=split_config)


def test_dummy_env_split_actions_continuous() -> None:
    """Test for continuous actions"""

    split_config = {
        'action_0_2': {
            'action_up': {
                'indices': [0, 2]
            },
            'action_side': {
                'indices': [1, 3, 4]
            }
        }
    }
    _test_dummy_env_for_split_config(split_config)


def test_dummy_env_split_actions_multi_discrete() -> None:
    """Test for multi-discrete actions."""

    split_config = {
        'action_0_1': {
            'action_0_1-0': {
                'indices': [1]
            },
            'action_0_1-1': {
                'indices': [0]
            }
        }
    }
    _test_dummy_env_for_split_config(split_config)


def test_dummy_env_split_actions_multi_discrete_multi() -> None:
    """Test for multi-discrete actions resulting in multi-discrete actions"""

    split_config = {
        'action_0_1': {
            'action_0_1-rev': {
                'indices': [1, 0]
            }
        }
    }
    _test_dummy_env_for_split_config(split_config)


def test_dummy_env_split_actions_multi_binary() -> None:
    """Test for multi-binary actions"""

    split_config = {
        'action_1_1': {
            'action_1_1-0': {
                'indices': [1]
            },
            'action_1_1-1': {
                'indices': [0, 2, 3, 4]
            }
        }
    }
    _test_dummy_env_for_split_config(split_config)
