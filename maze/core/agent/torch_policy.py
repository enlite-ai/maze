"""Encapsulation of multiple torch policies for training and rollouts in structured environments."""
import dataclasses
from typing import Mapping, Union, Any, List, Dict, Tuple, Sequence, Optional

import torch
from torch import nn

from maze.core.agent.policy import Policy
from maze.core.agent.torch_model import TorchModel
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType, TorchActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID, StepKeyType
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.distributions.dict import DictProbabilityDistribution
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.perception_utils import convert_to_torch, convert_to_numpy


@dataclasses.dataclass
class PolicySubStepOutput:
    """ Dataclass for holding the output of the policy's compute full output method """

    action_logits: Dict[str, torch.Tensor]
    """A logits dictionary [action_head: action_logits] to parameterize the distribution from."""

    prob_dist: DictProbabilityDistribution
    """The respective instance of a DictProbabilityDistribution."""

    # TODO
    deterministic: bool
    """ : """

    embedding_logits: Optional[Dict[str, torch.Tensor]]
    """The Embedding output if applicable, used as the input for the critic network."""

    @property
    def entropy(self) -> torch.Tensor:
        """The entropy of the probability distribution."""
        return self.prob_dist.entropy()

    # @property
    # def sampled_actions(self) -> Dict[str, torch.Tensor]:
    #     """The samples actions for the computed logits w.r.t. the distribution, temperature and determinism."""
    #     if self.deterministic:
    #         return self.prob_dist.deterministic_sample()
    #     else:
    #         return self.prob_dist.sample()

    # def compute_action_log_porbs(self, sampled_actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     return self.prob_dist.log_prob(sampled_actions)


class PolicyOutput:
    """TODO:"""
    def __init__(self):
        self._step_policy_outputs: Dict[StepKeyType, PolicySubStepOutput] = dict()

    def __setitem__(self, key: StepKeyType, value: PolicySubStepOutput):
        """ Set self[key] to value. """
        self._step_policy_outputs[key] = value

    def keys(self):
        return self._step_policy_outputs.keys()

    @property
    def action_logits(self):
        return {step_key: self._step_policy_outputs[step_key].action_logits for step_key in self.keys()}

    @property
    def prob_dist(self):
        return {step_key: self._step_policy_outputs[step_key].prob_dist for step_key in self.keys()}

    @property
    def entropy(self):
        return {step_key: self._step_policy_outputs[step_key].entropy for step_key in self.keys()}

    # @property
    # def sampled_actions(self):
    #     return {step_key: self._step_policy_outputs[step_key].sampled_actions for step_key in self.keys()}

    @property
    def embedding_logits(self):
        return {step_key: self._step_policy_outputs[step_key].embedding_logits for step_key in self.keys()}

    # def compute_action_log_probs(self, sampled_actions: Dict[StepKeyType, Dict[str, torch.Tensor]]):
    #     return {step_key: self._step_policy_outputs[step_key].compute_action_log_porbs(sampled_actions[step_key]) for step_key
    #             in self.keys()}

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

    # Rollout methods: -------------------------------------------------------------------------------------------------

    @override(Policy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[MazeStateType] = None,
                       env: Optional[BaseEnv] = None,
                       actor_id: ActorID = None,
                       deterministic: bool = False) -> ActionType:
        """implementation of :class:`~maze.core.agent.policy.Policy`
        """
        action, _ = self.compute_action_with_logits(observation, actor_id, deterministic)
        return action

    @override(Policy)
    def compute_top_action_candidates(self,
                                      observation: ObservationType,
                                      num_candidates: Optional[int],
                                      maze_state: Optional[MazeStateType] = None,
                                      env: Optional[BaseEnv] = None,
                                      actor_id: ActorID = None,
                                      deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """implementation of :class:`~maze.core.agent.policy.Policy`"""
        raise NotImplementedError

    # TODO: Remove this method, since it only is used by impala actor rollout
    def compute_action_with_logits(self, observation: Any, actor_id: ActorID = None,
                                   deterministic: bool = False) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Compute action for the given observation and policy ID and return it together with the logits.

        :param observation: Current observation of the environment
        :param actor_id: ID of the actor (does not have to be provided if policies dict contain only 1 policy)
        :param deterministic: Specify if the action should be computed deterministically
        :return: Tuple of (action, logits_dict)
        """
        with torch.no_grad():
            logits_dict, _ = self.compute_logits_dict(observation, actor_id)
            prob_dist = self.logits_dict_to_distribution(logits_dict)
            if deterministic:
                sampled_action = prob_dist.deterministic_sample()
            else:
                sampled_action = prob_dist.sample()

        return convert_to_numpy(sampled_action, cast=None, in_place=False), logits_dict

    # Learner methods: -------------------------------------------------------------------------------------------------

    # TODO: used by sac trainer
    def compute_action_logits_entropy_dist(
            self, actor_id: ActorID, observation: Dict[Union[str, int], torch.Tensor],
            deterministic: bool, temperature: float) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor],
                                                              torch.Tensor, DictProbabilityDistribution]:
        """Compute action for the given observation and policy ID and return it together with the logits.

        :param actor_id: ID of the actor to query a policy for
        :param observation: Current observation of the environment
        :param deterministic: Specify if the action should be computed deterministically
        :param temperature: Controls the sampling behaviour.
                                * 1.0 corresponds to unmodified sampling
                                * smaller than 1.0 concentrates the action distribution towards deterministic sampling
        :return: Tuple of (action, logits_dict, entropy, prob_dist)
        """

        obs_t = convert_to_torch(observation, device=self._device, cast=None, in_place=True)
        logits_dict, _ = self.compute_logits_dict(obs_t, actor_id)
        prob_dist = self.logits_dict_to_distribution(logits_dict, temperature)
        if deterministic:
            sampled_action = prob_dist.deterministic_sample()
        else:
            sampled_action = prob_dist.sample()

        return sampled_action, logits_dict, prob_dist.entropy(), prob_dist

    # TODO: This method is only used by mcts policy !!! can be deleted
    def compute_action_distribution(self, observation: Any, actor_id: ActorID = None) -> Any:
        """Query the policy corresponding to the given ID for the action distribution.

        :param observation: Observation to get action distribution for
        :param actor_id: Actor ID corresponding to the observation
        :return: Action distribution for the given observation
        """
        logits_dict, _ = self.compute_logits_dict(observation, actor_id)
        return self.distribution_mapper.logits_dict_to_distribution(logits_dict=logits_dict, temperature=1.0)

    # TODO: this is used in the torch policy, custom/template model composer (to get the desired shape) will be removed,
    # todo: bc loss, impala learner, actor_critic_learner
    def compute_logits_dict(self, observation: Any, actor_id: ActorID = None,
                            return_embedding: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get the logits for the given observation and actor ID.

        :param observation: Observation to return probability distribution for
        :param actor_id: Actor ID this observation corresponds to
        :param return_embedding: Specify whether to return the embedding output of the policy network.
        :return: Tuple of Logits dictionary and embedding output if return embedding is set to true.
        """
        obs_t = convert_to_torch(observation, device=self._device, cast=None, in_place=True)
        network_out = self.network_for(actor_id)(obs_t)
        # action_logits, embedding

        embedding_out = None
        if return_embedding:
            embedding_out = dict(filter(lambda ii: ii[0] not in self.distribution_mapper.action_space.spaces.keys(),
                                        network_out.items()))
            assert len(embedding_out) > 0, 'If return embedding is specified the network should also produce an ' \
                                           'embedding output.'

        if any([key not in self.distribution_mapper.action_space.spaces.keys() for key in
                network_out.keys()]):
            network_out = dict(filter(lambda ii: ii[0] in self.distribution_mapper.action_space.spaces.keys(),
                                      network_out.items()))

        return network_out, embedding_out

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


    # TODO: used by: torch policy, bc loss, rllib trainer, actorcritic trainer !!!! can be deleted
    def logits_dict_to_distribution(self, logits_dict: Dict[str, torch.Tensor], temperature: float = 1.0):
        """Helper function for creation of a dict probability distribution from the given logits dictionary.

        :param logits_dict: A logits dictionary [action_head: action_logits] to parameterize the distribution from.
        :param temperature: Controls the sampling behaviour.
                                * 1.0 corresponds to unmodified sampling
                                * smaller than 1.0 concentrates the action distribution towards deterministic sampling
        :return: (DictProbabilityDistribution) the respective instance of a DictProbabilityDistribution.
        """
        return self.distribution_mapper.logits_dict_to_distribution(logits_dict, temperature)

    # METHODS TO KEEP

    @staticmethod
    def sample_action_from_prob_dist(policy_output: PolicyOutput, deterministic: bool):
        """TODO:"""
        if deterministic:
            return {kk: vv.deterministic_sample() for kk, vv in policy_output.prob_dist.items()}
        else:
            return {kk: vv.sample() for kk, vv in policy_output.prob_dist.items()}

    @staticmethod
    def compute_action_log_probs(policy_output: PolicyOutput, actions: Dict[StepKeyType, TorchActionType]):
        """TODO"""
        return {kk: vv.log_prob(actions[kk]) for kk, vv in policy_output.prob_dist.items()}

    def compute_substep_policy_output(self, observation: ObservationType, actor_id: ActorID = None,
                                      temperature: float = 1.0, deterministic: bool = False) -> PolicySubStepOutput:
        """TODO:"""

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
                                   deterministic=deterministic)

    def compute_policy_output(self, record: StructuredSpacesRecord, temperature: float = 1.0,
                              deterministic: bool = False) -> PolicyOutput:
        """TODO"""

        structured_policy_output = PolicyOutput()
        for substep_record in record.substep_records:
            structured_policy_output[substep_record.substep_key] = self.compute_substep_policy_output(
                substep_record.observation, actor_id=substep_record.actor_id, temperature=temperature,
                deterministic=deterministic)
        return structured_policy_output
