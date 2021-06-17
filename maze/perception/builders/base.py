""" Contains interfaces for perception model builders. """
from abc import abstractmethod, ABC
from typing import Dict, Union, Any, Optional, List, Iterable

import numpy as np
from gym import spaces
from omegaconf import ListConfig, DictConfig

from maze.core.env.structured_env import StepKeyType
from maze.perception.blocks.inference import InferenceBlock


class BaseModelBuilder(ABC):
    """Base class for perception default model builders.

    :param modality_config: dictionary mapping perception modalities to blocks and block config parameters.
    :param observation_modality_mapping: A mapping of observation keys to perception modalities.
    :param shared_embedding_keys: The shared embedding keys to use as an input to the critic network where the value
        can be one of the following:
        - None, empty list or dict of empty lists: No shared embeddings are used.
        - A list of str values: The shared embedding keys to use for creating the critic network in each substep
        (the same keys).
        - A dict of lists: Here the keys have to refer to the step-keys of the environment, the corresponding lists
        specify the input keys to the critic network in this step.
    """

    def __init__(self, modality_config: Dict[str, Union[str, Dict[str, Any]]],
                 observation_modality_mapping: Dict[str, str],
                 shared_embedding_keys: Optional[Union[List[str], Dict[str, List[str]]]]):
        self.modality_config = modality_config
        self.observation_modality_mapping = observation_modality_mapping

        # Init shared embedding info
        self.shared_embedding_keys = shared_embedding_keys
        self.use_shared_embedding = None

    def init_shared_embedding_keys(self, step_keys: Iterable[StepKeyType]) -> None:
        """Init the shared embedding keys as a dict using the step_keys of the environment.

        :param step_keys: the step keys of the environment steps
        """
        # Init shared embedding keys
        self.shared_embedding_keys = list() if self.shared_embedding_keys is None else self.shared_embedding_keys
        if isinstance(self.shared_embedding_keys, (list, ListConfig)):
            self.shared_embedding_keys = {step_key: list(self.shared_embedding_keys) for step_key in step_keys}
        else:
            assert isinstance(self.shared_embedding_keys, (dict, DictConfig)), f'type: ' \
                                                                                f'{type(self.shared_embedding_keys)}'
            self.shared_embedding_keys = {step_key: list(shared_keys) for step_key, shared_keys in
                                          self.shared_embedding_keys.items()}
        self.use_shared_embedding: Dict[StepKeyType, bool] = {step_key: len(shared_keys) > 0 for step_key, shared_keys
                                                              in self.shared_embedding_keys.items()}

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
