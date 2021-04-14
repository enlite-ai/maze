"""Composer implementation for step critic."""
from typing import Dict, Union

from gym import spaces
from torch import nn

from maze.core.agent.torch_state_critic import TorchStepStateCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StepKeyType
from maze.core.utils.config_utils import list_to_dict
from maze.core.utils.factory import CollectionOfConfigType, Factory
from maze.perception.models.critics.base_state_critic_composer import BaseStateCriticComposer


class StepStateCriticComposer(BaseStateCriticComposer):
    """Each sub-step or actor gets its individual critic.

    Instantiates a
    :class:`~maze.core.agent.torch_state_critic.TorchStepStateCritic`.

    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param networks: Critics networks as defined in the config (either list or dictionary of object params and type).
    """

    def __init__(self,
                 observation_spaces_dict: Dict[Union[str, int], spaces.Dict],
                 agent_counts_dict: Dict[StepKeyType, int],
                 networks: CollectionOfConfigType):
        super().__init__(observation_spaces_dict, agent_counts_dict)

        # initialize critics
        networks = list_to_dict(networks)
        self._critics = {key: Factory(base_type=nn.Module).instantiate(networks[key],
                                                                       obs_shapes=self._obs_shapes[key])
                         for key in networks.keys()}

    @property
    @override(BaseStateCriticComposer)
    def critic(self) -> TorchStepStateCritic:
        """implementation of :class:`~maze.perception.models.critics.base_state_critic_composer.BaseStateCriticComposer`
        """
        return TorchStepStateCritic(self._critics, num_policies=len(self._obs_shapes), device="cpu")
