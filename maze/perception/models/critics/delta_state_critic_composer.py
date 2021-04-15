"""Composer implementation for delta critic."""
from typing import Dict, Union

import numpy as np
from gym import spaces
from torch import nn

from maze.core.agent.torch_state_critic import TorchDeltaStateCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StepKeyType
from maze.core.utils.config_utils import list_to_dict
from maze.core.utils.factory import CollectionOfConfigType, Factory
from maze.perception.models.critics.base_state_critic_composer import BaseStateCriticComposer


class DeltaStateCriticComposer(BaseStateCriticComposer):
    """First sub step gets a regular critic, subsequent sub-steps predict a delta w.r.t. to the previous critic.

    Instantiates a
    :class:`~maze.core.agent.torch_state_critic.TorchDeltaStateCritic`.

    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param networks: The single, shared critic network as defined in the config.
    """
    prev_value_key = 'prev_value'
    prev_value_shape = (1,)
    prev_value_space = spaces.Dict({prev_value_key: spaces.Box(0, 1, shape=prev_value_shape, dtype=np.float32)})

    def __init__(self,
                 observation_spaces_dict: Dict[Union[str, int], spaces.Dict],
                 agent_counts_dict: Dict[StepKeyType, int],
                 networks: CollectionOfConfigType):
        super().__init__(observation_spaces_dict, agent_counts_dict)

        # initialize critic
        model_registry = Factory(base_type=nn.Module)
        networks = list_to_dict(networks)
        self._critics = dict()
        for idx, (key, net_config) in enumerate(networks.items()):
            step_obs_shapes = self._obs_shapes[key]
            if idx > 0:
                step_obs_shapes = {**step_obs_shapes, self.prev_value_key: self.prev_value_shape}
            self._critics[key] = model_registry.instantiate(networks[key], obs_shapes=step_obs_shapes)

    @property
    @override(BaseStateCriticComposer)
    def critic(self) -> TorchDeltaStateCritic:
        """implementation of :class:`~maze.perception.models.critics.base_state_critic_composer.BaseStateCriticComposer`
        """
        return TorchDeltaStateCritic(self._critics, num_policies=len(self._obs_shapes), device="cpu")
