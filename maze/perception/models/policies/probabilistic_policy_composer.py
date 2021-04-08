"""Composer for probabilistic policy (actor) networks."""
from typing import Dict, Union

from gym import spaces
from torch import nn

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.structured_env import StepKeyType
from maze.core.utils.config_utils import list_to_dict
from maze.core.utils.factory import CollectionOfConfigType, Factory
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
                 action_spaces_dict: Dict[StepKeyType, spaces.Dict],
                 observation_spaces_dict: Dict[StepKeyType, spaces.Dict],
                 agent_counts_dict: Dict[StepKeyType, int],
                 separated_agent_networks: bool,
                 distribution_mapper: DistributionMapper,
                 networks: CollectionOfConfigType):
        super().__init__(action_spaces_dict, observation_spaces_dict, agent_counts_dict, distribution_mapper)
        self._separated_agent_networks = separated_agent_networks

        # initialize policies
        self._networks = list_to_dict(networks)

        if self._separated_agent_networks:
            # Build a separate network for each sub-step and agent
            self._policies = {}
            for substep_key in self._networks.keys():
                for agent_id in range(agent_counts_dict[substep_key]):
                    self._policies[(substep_key, agent_id)] = self._network_for_substep_key(substep_key)
        else:
            # Build a network for each sub-steps, with all agents for this sub-step sharing it
            self._policies = {key: self._network_for_substep_key(key) for key in self._networks.keys()}

    @property
    def policy(self) -> TorchPolicy:
        """implementation of :class:`~maze.perception.models.policies.base_policy_composer.BasePolicyComposer`
        """
        return TorchPolicy(networks=self._policies,
                           distribution_mapper=self._distribution_mapper,
                           separated_agent_networks=self._separated_agent_networks,
                           device='cpu')

    def _network_for_substep_key(self, key: StepKeyType):
        return Factory(nn.Module).instantiate(self._networks[key],
                                              obs_shapes=self._obs_shapes[key],
                                              action_logits_shapes=self._action_logit_shapes[key])
