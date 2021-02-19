"""Composer for probabilistic policy (actor) networks."""
from typing import Dict, Union

from gym import spaces
from torch import nn

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.utils.config_utils import list_to_dict
from maze.core.utils.registry import CollectionOfConfigType, Registry
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.models.policies.base_policy_composer import BasePolicyComposer


class ProbabilisticPolicyComposer(BasePolicyComposer):
    """Composes networks for probabilistic policies.

    :param action_spaces_dict: Dict of sub-step id to action space.
    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param distribution_mapper: The distribution mapper.
    :param networks: Policy networks as defined in the config (either list or dictionary of object params and type).
    """

    def __init__(self,
                 action_spaces_dict: Dict[Union[str, int], spaces.Dict],
                 observation_spaces_dict: Dict[Union[str, int], spaces.Dict],
                 distribution_mapper: DistributionMapper,
                 networks: CollectionOfConfigType):
        super().__init__(action_spaces_dict, observation_spaces_dict, distribution_mapper)

        # initialize policies
        networks = list_to_dict(networks)
        model_registry = Registry(base_type=nn.Module)

        self._policies = {key: model_registry.arg_to_obj(networks[key],
                                                         obs_shapes=self._obs_shapes[key],
                                                         action_logits_shapes=self._action_logit_shapes[key])
                          for key in networks.keys()}

    @property
    def policy(self) -> TorchPolicy:
        """implementation of :class:`~maze.perception.models.policies.base_policy_composer.BasePolicyComposer`
        """
        return TorchPolicy(networks=self._policies,
                           distribution_mapper=self._distribution_mapper,
                           device='cpu')
