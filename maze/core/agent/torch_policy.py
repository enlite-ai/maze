"""Encapsulation of multiple torch policies for training and rollouts in structured environments."""
from typing import Mapping, Union, List, Dict, Tuple, Sequence, Optional

import torch
from torch import nn

from maze.core.agent.policy import Policy
from maze.core.agent.torch_model import TorchModel
from maze.core.agent.torch_policy_output import PolicySubStepOutput, PolicyOutput
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID, StepKeyType
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.distributions.categorical import CategoricalProbabilityDistribution
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.perception_utils import convert_to_torch, convert_to_numpy


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
                 device: str,
                 substeps_with_separate_agent_nets: Optional[List[StepKeyType]] = None):
        self.networks = networks
        self.distribution_mapper = distribution_mapper

        if substeps_with_separate_agent_nets is not None:
            self.substeps_with_separate_agent_nets = set(substeps_with_separate_agent_nets)
        else:
            self.substeps_with_separate_agent_nets = set()

        TorchModel.__init__(self, device=device)

    @override(Policy)
    def seed(self, seed: int) -> None:
        """This is done globally"""
        pass

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state() object to compute the action."""
        return False

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

    @override(Policy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[MazeStateType] = None,
                       env: Optional[BaseEnv] = None,
                       actor_id: ActorID = None,
                       deterministic: bool = False) -> ActionType:
        """implementation of :class:`~maze.core.agent.policy.Policy`
        """
        with torch.no_grad():
            policy_out = self.compute_substep_policy_output(observation, actor_id)
            if deterministic:
                action = policy_out.prob_dist.deterministic_sample()
            else:
                action = policy_out.prob_dist.sample()
        return convert_to_numpy(action, cast=None, in_place=False)

    @override(Policy)
    def compute_top_action_candidates(self, observation: ObservationType, num_candidates: Optional[int],
                                      maze_state: Optional[MazeStateType], env: Optional[BaseEnv],
                                      actor_id: ActorID = None) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """implementation of :class:`~maze.core.agent.policy.Policy`"""
        with torch.no_grad():
            policy_out = self.compute_substep_policy_output(observation, actor_id)
            actions = list()
            probs = list()
            for k, dist in policy_out.prob_dist.distribution_dict.items():
                if isinstance(dist, CategoricalProbabilityDistribution):
                    for aa in dist.logits.argsort(-1, descending=True)[:num_candidates]:
                        actions.append({k: aa.numpy()})
                        probs.append(dist.dist.probs[aa])
                else:
                    raise NotImplementedError('Compute top action candidates only supports discrete action in torch '
                                              'policy')
        convert_to_numpy(actions, cast=None, in_place=False)
        return actions, probs

    def network_for(self, actor_id: Optional[ActorID]) -> nn.Module:
        """Helper function for returning a network for the given policy ID (using either just the sub-step ID
        or the full Actor ID as key, depending on the separated agent networks mode.

        :param actor_id: Actor ID to get a network for
        :return: Network corresponding to the given policy ID.
        """
        if actor_id is None:
            assert len(self.networks) == 1, "multiple networks are available, please specify the actor ID explicitly"
            return list(self.networks.values())[0]

        network_key = actor_id if actor_id.step_key in self.substeps_with_separate_agent_nets else actor_id.step_key
        return self.networks[network_key]

    def compute_substep_policy_output(self, observation: ObservationType, actor_id: ActorID = None,
                                      temperature: float = 1.0) -> PolicySubStepOutput:
        """Compute the full output of a specified policy.

        :param observation: The observation to use as input.
        :param actor_id: Optional, the actor id specifying the network to use.
        :param temperature: Optional, the temperature to use for initializing the probability distribution.
        :return: The computed PolicySubStepOutput.
        """

        # Convert the import to torch in-place
        obs_t = convert_to_torch(observation, device=self._device, cast=None, in_place=True)

        # Compute a forward pass of the policy network retrieving all the outputs (action logits + embedding logits if
        #  applicable)
        network_out = self.network_for(actor_id)(obs_t)

        # Disentangle action and embedding logits
        if any([key not in self.distribution_mapper.action_space.spaces.keys() for key in
                network_out.keys()]):

            # Filter out the action logits
            action_logits = dict(filter(lambda ii: ii[0] in self.distribution_mapper.action_space.spaces.keys(),
                                        network_out.items()))
            # Filter out the embedding logits
            embedding_logits = dict(filter(lambda ii: ii[0] not in self.distribution_mapper.action_space.spaces.keys(),
                                           network_out.items()))
        else:
            action_logits = network_out
            embedding_logits = None

        # Initialize the probability distributions
        prob_dist = self.distribution_mapper.logits_dict_to_distribution(action_logits, temperature)

        return PolicySubStepOutput(action_logits=action_logits, prob_dist=prob_dist, embedding_logits=embedding_logits,
                                   actor_id=actor_id)

    def compute_policy_output(self, record: StructuredSpacesRecord, temperature: float = 1.0) -> PolicyOutput:
        """Compute the full Policy output for all policy networks over a full (flat) environment step.

        :param record: The StructuredSpacesRecord holding the observation and actor ids.
        :param temperature: Optional, the temperature to use for initializing the probability distribution.
        :return: The full Policy output for the record given.
        """

        structured_policy_output = PolicyOutput()
        for substep_record in record.substep_records:
            structured_policy_output.append(self.compute_substep_policy_output(
                substep_record.observation, actor_id=substep_record.actor_id, temperature=temperature))
        return structured_policy_output
