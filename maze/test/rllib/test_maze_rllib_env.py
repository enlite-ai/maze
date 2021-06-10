"""Test maze rllib env"""
from collections import Callable

import pytest
from gym.envs.classic_control import CartPoleEnv

from maze.rllib.maze_rllib_env import build_maze_rllib_env_factory
from maze.test.rllib.test_that_segfault_has_been_fixed import assert_that_patch_has_been_applied
from maze.test.shared_test_utils.hydra_helper_functions import load_hydra_config


@pytest.mark.rllib
def test_maze_rllib_env():
    hydra_overrides = {}

    cfg = load_hydra_config('maze.conf', 'conf_rllib', hydra_overrides)

    env_factory = build_maze_rllib_env_factory(cfg)

    assert isinstance(env_factory, Callable)

    env = env_factory()
    assert isinstance(env, CartPoleEnv)

    assert_that_patch_has_been_applied()
