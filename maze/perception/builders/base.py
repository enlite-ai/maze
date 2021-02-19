""" Contains interfaces for perception model builders. """
from abc import abstractmethod, ABC
from typing import Dict, Union, Any

import numpy as np
from gym import spaces

from maze.perception.blocks.inference import InferenceBlock


class BaseModelBuilder(ABC):
    """Base class for perception default model builders.

    :param: modality_config: dictionary mapping perception modalities to blocks and block config parameters.
    :param observation_modality_mapping: A mapping of observation keys to perception modalities.
    """

    def __init__(self, modality_config: Dict[str, Union[str, Dict[str, Any]]],
                 observation_modality_mapping: Dict[str, str]):
        self.modality_config = modality_config
        self.observation_modality_mapping = observation_modality_mapping

    @classmethod
    def to_recurrent_gym_space(cls, observation_space: spaces.Dict, rnn_steps: int) -> spaces.Dict:
        """Converts the given observation space to a recurrent space.

        :param observation_space: The respective observation space.
        :param rnn_steps: Number of recurrent time steps.
        :return: The rnn modified dictionary observation space.
        """
        assert rnn_steps > 1

        rnn_dict = dict()
        for key, space in observation_space.spaces.items():
            assert isinstance(space, spaces.Box)
            rnn_low = np.repeat(space.low[np.newaxis], axis=0, repeats=rnn_steps)
            rnn_high = np.repeat(space.high[np.newaxis], axis=0, repeats=rnn_steps)
            rnn_dict[key] = spaces.Box(low=rnn_low, high=rnn_high, dtype=space.dtype)

        return spaces.Dict(rnn_dict)

    @abstractmethod
    def from_observation_space(self, observation_space: spaces.Dict) -> InferenceBlock:
        """Compiles an inference graph for a given observation space.

        Only observations which are contained in the self.observation_modalities dictionary are considered.

        :param observation_space: The respective observation space.
        :return: the resulting inference block.
        """
