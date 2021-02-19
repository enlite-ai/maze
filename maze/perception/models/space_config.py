"""Configuration of environment spaces (action & observation) used for model config."""

import pickle
from typing import Union, Dict

import gym


class SpacesConfig:
    """Represents configuration of environment spaces (action & observation) used for model config.

    Spaces config are needed (together with model config and dumped state dict) when loading
    a trained policy for rollout.
    """
    def __init__(self,
                 action_spaces_dict: Dict[Union[str, int], gym.spaces.Dict],
                 observation_spaces_dict: Dict[Union[str, int], gym.spaces.Dict]):
        self.action_spaces_dict = action_spaces_dict
        self.observation_spaces_dict = observation_spaces_dict

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
