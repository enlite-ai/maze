"""Utility methods used throughout the code base"""
import os
from pathlib import Path
from typing import Mapping, Union, Sequence

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.experimental import initialize_config_module, compose

from maze.core.env.maze_env import MazeEnv
from maze.core.utils.factory import Factory, ConfigType, CollectionOfConfigType
from maze.core.wrappers.wrapper_factory import WrapperFactory


def read_config(path: Union[Path, str]) -> dict:
    """
    Read YAML file into a dict

    :param path: Path of the file to read
    :return: Dict with the YAML file contents
    """
    with open(str(path), 'r') as in_config:
        config = yaml.safe_load(in_config)
    return config


def list_to_dict(list_or_dict: Union[list, Mapping]) -> Mapping:
    """Convert lists to int-indexed dicts.

    Code is simplified by supporting only one universal data structure instead of implementing code paths for lists
    and dicts separately.

    :param list_or_dict: The list to convert to dict. If it is already a dict,
                         the dict is returned without modification.
    :return: The passed list as dict.
    """
    if isinstance(list_or_dict, Mapping):
        return list_or_dict

    return {i: s for i, s in enumerate(list_or_dict)}


def int_range(stop: int) -> Sequence:
    """Simple wrapper around builtin.range which can be used in Hydra yaml configs"""
    return range(stop)


class EnvFactory:
    """Helper class to instantiate an environment from configuration with the help of the Registry.

    :param env: environment configuration
    :param wrappers: collection of wrappers as configuration
    """

    def __init__(self, env: ConfigType, wrappers: CollectionOfConfigType):
        self.env = env
        self.wrappers = wrappers

    def __call__(self, *args, **kwargs) -> MazeEnv:
        """environment factory
        :return: Newly created environment instance.
        """
        env = Factory(MazeEnv).instantiate(self.env)
        env = WrapperFactory.wrap_from_config(env, self.wrappers)

        return env


def make_env(env: ConfigType, wrappers: CollectionOfConfigType) -> MazeEnv:
    """Helper to create a single environment from configuration"""
    env_factory = EnvFactory(env=env, wrappers=wrappers)

    return env_factory()


def make_env_from_hydra(config_module: str,
                        config_name: str = None,
                        **hydra_overrides: str) -> MazeEnv:
    """Create an environment instance from the hydra configuration, given the overrides.
    :param config_module: Python module path of the hydra configuration package
    :param config_name: Name of the defaults configuration yaml within `config_module`
    :param hydra_overrides: Overrides as kwargs, e.g. env="cartpole", configuration="test"
    :return: The newly instantiated environment
    """
    with initialize_config_module(config_module):
        # config is relative to a module
        cfg = compose(config_name, overrides=[key + "=" + value for key, value in hydra_overrides.items()])
        env_factory = EnvFactory(cfg.env, cfg.wrappers if "wrappers" in cfg else {})
        return env_factory()


class SwitchWorkingDirectoryToInput:
    """
    Hydra is configured to create a fresh output directory for each run.
    However, to ensure model states, normalization stats and else are loaded from expected
    locations, we will change the dir back to the original working dir for the initialization
    (and then change it back so that all later script output lands in the hydra output dir as expected)
    """

    def __init__(self, input_dir: str):
        self.input_dir = input_dir

    def __enter__(self):
        # do nothing if hydra is not initialized
        if not HydraConfig.initialized() or not self.input_dir:
            return

        self.hydra_out_dir = os.getcwd()
        original_dir = os.path.join(hydra.utils.get_original_cwd(), self.input_dir)
        print(f"Switching load directory to {original_dir}")
        os.chdir(original_dir)

    def __exit__(self, *args):
        # do nothing if hydra is not initialized
        if not HydraConfig.initialized() or not self.input_dir:
            return

        os.chdir(self.hydra_out_dir)
