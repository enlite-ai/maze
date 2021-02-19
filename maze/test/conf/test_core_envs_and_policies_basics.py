"""Test standard tasks like instantiation and random sampling on all discovered envs."""
from typing import Dict

import pytest

from maze.test.shared_test_utils.hydra_helper_functions import get_all_configs_from_hydra, check_random_sampling, \
    check_env_and_model_instantiation


@pytest.mark.parametrize("config_module, config, overrides", get_all_configs_from_hydra("conf_rollout", ["maze.conf"]))
def test_instantiation(config_module: str, config: str, overrides: Dict[str, str]):
    """test if the envs can be instantiated"""
    check_env_and_model_instantiation(config_module, config, overrides)


@pytest.mark.parametrize("config_module, config, overrides", get_all_configs_from_hydra("conf_rollout", ["maze.conf"]))
def test_random_sampling(config_module: str, config: str, overrides: Dict[str, str]):
    """ tests random sampling in hydra configured environments. """
    check_random_sampling(config_module, config, overrides)
