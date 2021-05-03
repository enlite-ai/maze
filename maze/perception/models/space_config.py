"""Configuration of environment spaces (action & observation) used for model config."""

import pickle
from typing import Union, Dict, Any

import gym
from omegaconf import DictConfig

from maze.core.env.structured_env import StepKeyType


class SpacesConfig:
    """Represents configuration of environment spaces (action & observation) used for model config.

    Spaces config are needed (together with model config and dumped state dict) when loading
    a trained policy for rollout.
    """
    def __init__(self,
                 action_spaces_dict: Dict[StepKeyType, gym.spaces.Dict],
                 observation_spaces_dict: Dict[StepKeyType, gym.spaces.Dict],
                 agent_counts_dict: Dict[StepKeyType, int]):
        self.action_spaces_dict = action_spaces_dict
        self.observation_spaces_dict = observation_spaces_dict
        self.agent_counts_dict = agent_counts_dict

    def save(self, dump_file_path: str) -> None:
        """Save the spaces config to a file.

        :param dump_file_path: Where to save the spaces config.
        """
        with open(dump_file_path, "wb") as out_f:
            pickle.dump(self, out_f)

    @classmethod
    def load(cls, in_file_path: str) -> 'SpacesConfig':
        """Load a saved spaces config from a file.

        :param in_file_path: Where to load the spaces config from.
        :return: Loaded spaces config object
        """
        with open(in_file_path, "rb") as in_f:
            spaces_config = pickle.load(in_f)
        return spaces_config

    def __getstate__(self) -> Dict[str, Any]:
        """
        Return internal state for serialization.
        This is a workaround for unpickle-able objects in DictConfig's _parent node. Ideally this should be made
        redundant by having .instantiate return dicts instead of DictConfigs or any other standardized approach to
        converting DictConfigs as returned from Factory.instantiate() to native dicts.
        :return: Internal state as dictionary.
        """

        # Use Hydra's __get__() to return actual spaces as state instead of Hydra's DictConfig/Node objecs.
        return {
            "action_spaces_dict": {key: val for key, val in self.action_spaces_dict.items()},
            "observation_spaces_dict": {key: val for key, val in self.observation_spaces_dict.items()},
            "agent_counts_dict": {key: val for key, val in self.agent_counts_dict.items()},
        }
