""" Contains hydra helper functions for testing. """
from typing import List, Dict

import pytest
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize_config_module, compose
from omegaconf import DictConfig
from torch import nn

from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.utils.config_utils import make_env_from_hydra, EnvFactory
from maze.core.utils.factory import Factory
from maze.core.wrappers.observation_normalization.observation_normalization_utils import obtain_normalization_statistics
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper
from maze.maze_cli import maze_run
from maze.perception.models.model_composer import BaseModelComposer


def _get_all_overrides_from_hydra() -> List[Dict[str, str]]:
    """Enumerate all environment configurations from Hydra

    Note that only configurations from the main config module are returned.
    I.e., if the main config module is `maze_envs.logistics.conf`, then, for envs, only envs listed
    under `maze_envs.logistics.conf.envs` are returned. (This is to ensure that if we are testing
    e.g. logistics envs, the other available environments do not get mixed in.)
    """
    config_sources = GlobalHydra.instance().config_loader().get_sources()
    main_config_source = list(filter(lambda source: source.provider == "main", config_sources))[0]
    overrides = []

    for env in main_config_source.list("env", results_filter=None):
        overrides.append(dict(env=env))

    if main_config_source.exists("env_configuration"):
        for env_configuration in main_config_source.list("env_configuration", results_filter=None):
            env, configuration = env_configuration.split('-')
            overrides.append(dict(env=env, configuration=configuration))

    return overrides


def get_all_configs_from_hydra(default_conf: str, all_hydra_config_modules: List[str]) -> List[pytest.param]:
    """Enumerate all environment configurations from the Hydra options.

    :param default_conf: The name of the default config
    :param all_hydra_config_modules: A list of hydra config modules
    :return A list of pytest.param objects, to be used with   @pytest.mark.parametrize
    """
    configs = []

    for config_module in all_hydra_config_modules:
        # setup Hydra for the given config module
        with initialize_config_module(config_module):
            # query all argument overrides for this config module
            for overrides in _get_all_overrides_from_hydra():
                # add a single combination of module and hydra arguments to the list
                config = pytest.param(config_module, default_conf, overrides, id="-".join(overrides.values()))
                configs.append(config)

    return configs


def check_env_instantiation(config_module: str, config: str, overrides: Dict[str, str]) -> None:
    """Check if env instantiation works."""
    env = make_env_from_hydra(config_module, config, **overrides)
    assert env is not None
    assert isinstance(env, StructuredEnv)


def check_env_and_model_instantiation(config_module: str, config: str, overrides: Dict[str, str]) -> None:
    """Check if env instantiation works."""
    with initialize_config_module(config_module):
        # config is relative to a module
        cfg = compose(config, overrides=[key + "=" + value for key, value in overrides.items()])

    env_factory = EnvFactory(cfg.env, cfg.wrappers if "wrappers" in cfg else {})
    env = env_factory()
    assert env is not None
    assert isinstance(env, (StructuredEnv, StructuredEnvSpacesMixin))

    if 'model' in overrides and overrides['model'] == 'rllib':
        return

    if 'model' in cfg:
        model_composer = Factory(BaseModelComposer).instantiate(
            cfg.model,
            action_spaces_dict=env.action_spaces_dict,
            observation_spaces_dict=env.observation_spaces_dict,
            agent_counts_dict=env.agent_counts_dict
        )
        for pp in model_composer.policy.networks.values():
            assert isinstance(pp, nn.Module)

        if model_composer.critic:
            for cc in model_composer.critic.networks.values():
                assert isinstance(cc, nn.Module)


def check_random_sampling(config_module: str, config: str, overrides: Dict[str, str]) -> None:
    """Check if random sampling in instantiated env works."""
    env = make_env_from_hydra(config_module, config, **overrides)

    # estimate normalization stats if required
    if isinstance(env, ObservationNormalizationWrapper):
        normalization_statistics = obtain_normalization_statistics(env=env, n_samples=100)
        env.set_normalization_statistics(normalization_statistics)

    env.reset()

    # run interaction loop
    n_steps = 100
    for step in range(n_steps):
        # sample random action
        action = env.action_space.sample()

        # take env step
        state, reward, done, info = env.step(action)
        if done:
            env.reset()


def load_hydra_config(config_module: str, config_name: str, hydra_overrides: Dict[str, str]) -> DictConfig:
    """Load a hydra config from a given config module + config name and additional hydra overrides.

    :param config_module: The config module that should be used
    :param config_name: The name of the config that should be used
    :param hydra_overrides: The hydra overrides that should be applied
    :return: A dict config of the created hydra config
    """
    with initialize_config_module(config_module=config_module):
        # Config is relative to a module
        # For the HydraConfig init below, we need the hydra key there as well (=> return_hydra_config=True)
        cfg = compose(config_name=config_name,
                      overrides=[key + "=" + str(val) for key, val in hydra_overrides.items()])

    return cfg


def run_maze_from_str(config_module: str, config_name: str, hydra_overrides: Dict[str, str]) -> DictConfig:
    """Load a hydra config from a given config module + config name and additional hydra overrides and start the exp

    :param config_module: The config module that should be used
    :param config_name: The name of the config that should be used
    :param hydra_overrides: The hydra overrides that should be applied
    :return: A dict config of the created hydra config
    """
    cfg = load_hydra_config(config_module, config_name, hydra_overrides)
    maze_run(cfg)

    return cfg
