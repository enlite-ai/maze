"""Encapsulation of multiple torch policies for training and rollouts in structured environments."""
from typing import Mapping, Union, Any, List, Dict, Tuple, Sequence, Optional

import torch
from maze.core.agent.policy import Policy
from maze.core.agent.torch_model import TorchModel
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from torch import nn

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.distributions.dict import DictProbabilityDistribution
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.perception_utils import convert_to_torch, convert_to_numpy
from torch import nn


class TorchPolicy(TorchModel, Policy):
    """Encapsulates multiple torch policies along with a distribution mapper for training and rollouts
    in structured environments.

    :param networks: Mapping of policy networks to encapsulate
    :param distribution_mapper: Distribution mapper associated with the policy mapping.
    :param device: Device the policy should be located on (cpu or cuda)
    """

    def __init__(self,
                 networks: Mapping[Union[str, int], nn.Module],
                 distribution_mapper: DistributionMapper,
                 device: str):
        self.networks = networks
        self.distribution_mapper = distribution_mapper

        TorchModel.__init__(self, device=device)

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state() object to compute the action."""
        return False

    @override(Policy)
    def compute_action(self, observation: ObservationType, maze_state: Optional[MazeStateType] = None,
                       policy_id: Union[str, int] = None, deterministic: bool = False) -> ActionType:
        """implementation of :class:`~maze.core.agent.policy.Policy`
        """
        action, _ = self.compute_action_with_logits(observation, policy_id, deterministic)
        return action

    @override(Policy)
    def compute_top_action_candidates(self, observation: ObservationType,
                                      num_candidates: int, maze_state: Optional[MazeStateType] = None,
                                      policy_id: Union[str, int] = None,
                                      deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """implementation of :class:`~maze.core.agent.policy.Policy`
        """
        raise NotImplementedError

    def compute_action_with_logits(self, observation: Any, policy_id: Union[str, int] = None,
                                   deterministic: bool = False) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Compute action for the given observation and policy ID and return it together with the logits.

        :param observation: Current observation of the environment
        :param policy_id: ID of the policy to query (does not have to be provided if policies dict contain only 1 policy
        :param deterministic: Specify if the action should be computed deterministically
        :return: Tuple of (action, logits_dict)
        """
        with torch.no_grad():
            logits_dict = self.compute_logits_dict(observation, policy_id)
            prob_dist = self.logits_dict_to_distribution(logits_dict)
            if deterministic:
                sampled_action = prob_dist.deterministic_sample()
            else:
                sampled_action = prob_dist.sample()

        return convert_to_numpy(sampled_action, cast=None, in_place=False), logits_dict

    def compute_action_logits_entropy_dist(
            self, policy_id: Union[str, int], observation: Dict[Union[str, int], torch.Tensor],
            deterministic: bool, temperature: float) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor],
                                                              torch.Tensor, DictProbabilityDistribution]:
        """Compute action for the given observation and policy ID and return it together with the logits.

        :param policy_id: ID of the policy to query (does not have to be provided if policies dict contain only 1 policy
        :param observation: Current observation of the environment
        :param deterministic: Specify if the action should be computed deterministically
        :param temperature: Controls the sampling behaviour.
                                * 1.0 corresponds to unmodified sampling
                                * smaller than 1.0 concentrates the action distribution towards deterministic sampling
        :return: Tuple of (action, logits_dict, entropy, prob_dist)
        """

        obs_t = convert_to_torch(observation, device=self._device, cast=None, in_place=True)
        logits_dict = self.compute_logits_dict(obs_t, policy_id)
        prob_dist = self.logits_dict_to_distribution(logits_dict, temperature)
        if deterministic:
            sampled_action = prob_dist.deterministic_sample()
        else:
            sampled_action = prob_dist.sample()

        return sampled_action, logits_dict, prob_dist.entropy(), prob_dist

    def compute_action_distribution(self, observation: Any, policy_id: Union[str, int] = None) -> Any:
        """Query the policy corresponding to the given ID for the action distribution."""
        logits_dict = self.compute_logits_dict(observation, policy_id)
        return self.distribution_mapper.logits_dict_to_distribution(logits_dict=logits_dict, temperature=1.0)

    def compute_logits_dict(self, observation: Any, policy_id: Union[str, int] = None) -> Dict[str, torch.Tensor]:
        """Get the logits for the given observation and policy ID.

        :param observation: Observation to return probability distribution for
        :param policy_id: Policy ID this observation corresponds to
        :return: Logits dictionary
        """
        obs_t = convert_to_torch(observation, device=self._device, cast=None, in_place=True)
        return self.networks[policy_id](obs_t)

    def logits_dict_to_distribution(self, logits_dict: Dict[str, torch.Tensor], temperature: float = 1.0):
        """Helper function for creation of a dict probability distribution from the given logits dictionary.

        :param logits_dict: A logits dictionary [action_head: action_logits] to parameterize the distribution from.
        :param temperature: Controls the sampling behaviour.
                                * 1.0 corresponds to unmodified sampling
                                * smaller than 1.0 concentrates the action distribution towards deterministic sampling
        :return: (DictProbabilityDistribution) the respective instance of a DictProbabilityDistribution.
        """
        return self.distribution_mapper.logits_dict_to_distribution(logits_dict, temperature)

    @override(TorchModel)
    def parameters(self) -> List[torch.Tensor]:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        params = []
        for policy in self.networks.values():
            params.extend(list(policy.parameters()))
        return params

    @override(TorchModel)
    def eval(self) -> None:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        for policy in self.networks.values():
            policy.eval()

    @override(TorchModel)
    def train(self) -> None:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        for policy in self.networks.values():
            policy.train()

    @override(TorchModel)
    def to(self, device: str) -> None:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        self._device = device
        for policy in self.networks.values():
            policy.to(device)

    @override(TorchModel)
    def state_dict(self) -> Dict:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        state_dict_policies = dict()

        for key, policy in self.networks.items():
            state_dict_policies[key] = policy.state_dict()

        return dict(policies=state_dict_policies)

    @override(TorchModel)
    def load_state_dict(self, state_dict: Dict) -> None:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        if "policies" in state_dict:
            state_dict_policies = state_dict["policies"]

            for key, policy in self.networks.items():
                assert key in state_dict_policies, f"Could not find state dict for policy ID: {key}"
                policy.load_state_dict(state_dict_policies[key])
