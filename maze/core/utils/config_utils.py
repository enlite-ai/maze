"""Utility methods used throughout the code base"""
import functools
import os
from pathlib import Path
from typing import Mapping, Union, Sequence

import hydra
import numpy as np
import yaml
from hydra import initialize_config_module, compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from packaging.version import Version

from maze.core.agent.serialized_torch_policy import SerializedTorchPolicy
from maze.core.env.maze_env import MazeEnv
from maze.core.utils.factory import Factory, ConfigType, CollectionOfConfigType
from maze.core.wrappers.wrapper_factory import WrapperFactory
from maze.utils.bcolors import BColors


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


def int_range_1000() -> Sequence:
    """Simple wrapper around builtin.range which can be used in Hydra yaml configs"""
    return range(1000)


def int_range_10000() -> Sequence:
    """Simple wrapper around builtin.range which can be used in Hydra yaml configs"""
    return range(10000)


def random_1000() -> Sequence:
    """Simple seeds generator which can be used in Hydra yaml configs"""
    rng = np.random.RandomState(1234)
    return [rng.randint(np.iinfo(np.int32).max) for _ in range(1000)]


def random_10000() -> Sequence:
    """Simple seeds generator which can be used in Hydra yaml configs"""
    rng = np.random.RandomState(1234)
    return [rng.randint(np.iinfo(np.int32).max) for _ in range(10000)]


def random_x_seeds(nb_seeds: int, repetitions: int) -> Sequence:
    """Simple seeds generator which can be used in Hydra yaml configs.

    :param nb_seeds: Number of seeds to generate.
    :param repetitions: Number of repetitions of each seed.

    :return: A sequence of random seeds (nb_seeds x repetitions).
    """
    rng = np.random.RandomState(1234)
    seeds = [rng.randint(np.iinfo(np.int32).max) for _ in range(nb_seeds)] * repetitions
    rng.shuffle(seeds)
    return seeds


def explicit_seed(seed: int, nb_seeds: int) -> Sequence[int]:
    """Return the same seed multiple times.

    :param seed: The seed to return.
    :param nb_seeds: The number of seeds to return.

    :return: The specified seed multiple times.
    """
    return [seed for _ in range(nb_seeds)]


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


def get_hydra_version_base() -> dict[str, str]:
    """Return hydra version base to use.

    :return: Hydra version base.
    """
    hydra_version = Version(hydra.__version__)
    assert hydra_version.major >= 1, f'Hydra major version must be at least >= 1; It is: {hydra_version}'

    if hydra_version.minor < 3:
        return {}

    return {'version_base': '1.1'}


def version_based_hydra_main(config_path: str, config_name: str):
    """Extend the hydra.main decorator >>@hydra.main(config_path="conf", config_name="conf_rollout")<< to add
    version_base if hydra version is above 1.2.xx.

    :param config_path: Path of the config file.
    :param config_name: Name of the config file.
    :return: The extended hydra.main decorator iff the hydra version is above 1.2.xx, otherwise the standard hydra.main
             deccorator.
    """
    def decorator(func):
        """Decorator that wraps a method.

        :param func: Method to wrap.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """The wrapper function that is called instead of the original function.

            :param args: Arguments passed to the original function.
            :param kwargs: Keyword arguments passed to the original function.
            :return: Wrapped function.
            """
            # Add additional hydra version base to hydra.main decorator
            hydra_version_base = get_hydra_version_base()
            return hydra.main(
                config_path=config_path,
                config_name=config_name,
                **hydra_version_base
            )(func)(*args, **kwargs)

        return wrapper

    return decorator


def read_hydra_config(config_module: str,
                      config_name: str = None,
                      **hydra_overrides: str) -> DictConfig:
    """Read and assemble a hydra config, given the config module, name, and overrides.

    :param config_module: Python module path of the hydra configuration package
    :param config_name: Name of the defaults configuration yaml file within `config_module`
    :param hydra_overrides: Overrides as kwargs, e.g. env="cartpole", configuration="test"
    :return: Hydra DictConfig instance, assembled according to the given module, name, and overrides.
    """
    # Add additional version_base iff hydra version is above 1.2.xx.
    kwargs = get_hydra_version_base()
    with initialize_config_module(config_module, **kwargs):
        cfg = compose(config_name, overrides=[key + "=" + value for key, value in hydra_overrides.items()])

    return cfg


def make_env_from_hydra(config_module: str,
                        config_name: str = None,
                        **hydra_overrides: str) -> MazeEnv:
    """Create an environment instance from the hydra configuration, given the overrides.
    :param config_module: Python module path of the hydra configuration package
    :param config_name: Name of the defaults configuration yaml within `config_module`
    :param hydra_overrides: Overrides as kwargs, e.g. env="cartpole", configuration="test"
    :return: The newly instantiated environment
    """
    cfg = read_hydra_config(config_module, config_name, **hydra_overrides)
    env_factory = EnvFactory(cfg.env, cfg.wrappers if "wrappers" in cfg else {})
    return env_factory()


class SwitchWorkingDirectoryToInput:
    """
    Context manager for temporarily switching directories (e.g., for loading policies or envs from output
    directories).

    Can be used also in the middle of a Hydra run, when Hydra already changed the current working directory,
    but the input_path is expected to be relative to the original one.

    More info:

    Hydra is configured to create a fresh output directory for each run.
    However, to ensure model states, normalization stats and else are loaded from expected
    locations, we will change the dir back to the original working dir for the initialization
    (and then change it back so that all later script output lands in the hydra output dir as expected)
    """

    def __init__(self, input_dir: str):
        self.input_dir = input_dir

    def __enter__(self):
        if not self.input_dir:
            return

        # we will return to current directory once the loading has been completed
        self.return_to_dir = os.getcwd()

        # if hydra is initialized, use the original working directory (before hydra changed it)
        if HydraConfig.initialized():
            original_work_dir = hydra.utils.get_original_cwd()
        else:
            original_work_dir = os.getcwd()
        input_dir_full_path = os.path.join(original_work_dir, self.input_dir)

        print(f"Switching load directory to {input_dir_full_path}")
        os.chdir(input_dir_full_path)

    def __exit__(self, *args):
        if not self.input_dir:
            return

        os.chdir(self.return_to_dir)


def make_env_from_output_dir(path: Union[Path, str], config_file: str) -> MazeEnv:
    """Create an environment instance from an output directory of a previous run. The
    directory is expected to contain the hydra config of the run and all associated
    information needed for the env (like observation normalization statistics).

    :param path: Path of the output directory to use
    :param config_file: The name of the config to read.
    :return: The newly instantiated environment
    """
    with SwitchWorkingDirectoryToInput(path):
        cfg = read_config(config_file)
        env = EnvFactory(cfg["env"], cfg["wrappers"] if "wrappers" in cfg else {})()
    return env


def make_policy_from_output_dir(path: Union[Path, str], config_file: str) -> SerializedTorchPolicy:
    """Create a serialized Torch policy instance from an output directory of a previous run. The
    directory is expected to contain the hydra config of the run and all associated
    information needed for the policy (like state_dict).

    :param path: Path of the output directory to use
    :param config_file: The name of the config to read.
    :return: The newly instantiated policy
    """
    with SwitchWorkingDirectoryToInput(path):
        cfg = read_config(config_file)
        policy = SerializedTorchPolicy(
            model=cfg["model"],
            state_dict_file="state_dict.pt",
            spaces_dict_file="spaces_config.pkl",
            device="cpu")

    return policy


def get_colored_config_str(cfg: DictConfig, resolve: bool) -> str:
    """Create and return a colord version of the config for printing or logging.

    :param cfg: The config of the experiment.
    :param resolve: Whether to resolve the config.
    :return: A colored string of the config.
    """
    # print and log config
    full_config = OmegaConf.to_container(cfg, resolve=resolve)

    runner_config = full_config.pop('runner')

    full_config_str = BColors.format_colored(yaml.dump({'env': full_config.pop('env')}, sort_keys=False),
                                             BColors.OKGREEN)
    full_config_str += BColors.format_colored(yaml.dump({'wrappers': full_config.pop('wrappers')}, sort_keys=False),
                                              BColors.OKGREEN)
    if 'model' in full_config:
        full_config_str += BColors.format_colored(yaml.dump({'model': full_config.pop('model')}), BColors.OKCYAN)

    # Blue config for algorithm or policy
    if 'algorithm' in full_config:
        full_config_str += BColors.format_colored(yaml.dump({'algorithm': full_config.pop('algorithm')}),
                                                  BColors.OKBLUE)
    elif 'policy' in full_config:
        full_config_str += BColors.format_colored(yaml.dump({'policy': full_config.pop('policy')}), BColors.OKBLUE)

    start_config_str = BColors.format_colored(yaml.dump({'runner': runner_config}), BColors.HEADER)
    start_config_str += BColors.format_colored(yaml.dump(full_config), BColors.HEADER)
    full_config_str = start_config_str + full_config_str

    return full_config_str
