"""
Various classes and helper functions used by RunContext and related classes.
"""
import enum
import os
from contextlib import contextmanager
from typing import Dict, Union, Mapping, Any


class RunMode(enum.Enum):
    """
    Available run modes for Python API.
    """

    ROLLOUT = "conf_rollout"
    TRAINING = "conf_train"


_OverridesType = Dict[str, Union[str, Mapping[str, Any], Any]]
"""Type for Hydra argument overrides."""
_PrimitiveType = Union[int, float, str, bool, enum.Enum]
"""Types considered as primitives."""
_MISSING_ARGUMENTS = {
    RunMode.TRAINING: {"policy"},
    RunMode.ROLLOUT: {"algorithm", "project.name", "critic"}
}
"""List of properties missing in run modes' default configurations."""
_ATTRIBUTE_PROXIES = {
    RunMode.TRAINING: {
        "policy": ("model", "policy"), "critic": ("model", "critic")
    },
    RunMode.ROLLOUT: {}
}
"""
Proxy keyword arguments don't reflect top-level properties of the underlying configuration, but act as shortcuts to 
subcomponents of top-level attributes (e.g.: "policy" is not a top-level attribute in training mode, but is 
shorthand for model.policy). Since they are more specific than potentially specified top-level arguments, they 
replace their equivalents in higher-level attributes. I.e.: If "model" and "policy" are specified, the "policy" 
value replaces the existing "model.policy".
"""


class RunContextError(Exception):
    """ Exception indicating Error in RunContext. """
    pass


class InvalidSpecificationError(RunContextError):
    """ Exception indicating Error due to inconsistent specification in RunContext. """
    pass


@contextmanager
def working_directory(path: str) -> None:
    """
    Switches working directory to specified path.
    :param path: Path to switch working directory to.
    """

    prev_cwd = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(prev_cwd)
