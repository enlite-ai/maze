"""Composer for probabilistic policy (actor) networks."""
from typing import Dict, List

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
                 distribution_mapper: DistributionMapper,
                 networks: CollectionOfConfigType,
                 substeps_with_separate_agent_nets: List[StepKeyType]):
        super().__init__(action_spaces_dict, observation_spaces_dict, agent_counts_dict, distribution_mapper)
        self._substeps_with_separate_agent_nets = substeps_with_separate_agent_nets

        # initialize policies
        self._networks = list_to_dict(networks)

        self._policies = {}
        for substep_key in self._networks.keys():

            # Build a separate network for each agent if configured
            if substep_key in substeps_with_separate_agent_nets:
                assert agent_counts_dict[substep_key] != -1, "for separated agent policies, the agent count needs to " \
                                                             "be known upfront"
                for agent_id in range(agent_counts_dict[substep_key]):
                    self._policies[(substep_key, agent_id)] = self._network_for_substep_key(substep_key)

            # Or a shared network common for all agents in this sub-step
            else:
                self._policies[substep_key] = self._network_for_substep_key(substep_key)

    @property
    def policy(self) -> TorchPolicy:
        """implementation of :class:`~maze.perception.models.policies.base_policy_composer.BasePolicyComposer`
        """
        return TorchPolicy(networks=self._policies, distribution_mapper=self._distribution_mapper, device='cpu',
                           substeps_with_separate_agent_nets=self._substeps_with_separate_agent_nets)

    def _network_for_substep_key(self, key: StepKeyType):
        return Factory(nn.Module).instantiate(self._networks[key],
                                              obs_shapes=self._obs_shapes[key],
                                              action_logits_shapes=self._action_logit_shapes[key])
