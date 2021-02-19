"""Contains utility functions for config testing."""
import os
from pathlib import Path

import yaml


def load_env_config(root_module, config_file: str) -> dict:
    """ Load yaml configuration used for testing.
    :param root_module: The root module holding the config file
    :param config_file: The name of the config file (e.g. "dummy_config_file.yml")
    :return: Environment configuration with specified wrappers.
    """
    module_path: str = list(root_module.__path__)[0]
    default_config_path: str = os.path.join(
        module_path, Path('.') / config_file
    )

    # load default config
    with open(default_config_path, 'r') as in_config:
        return yaml.safe_load(in_config)
