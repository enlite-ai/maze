"""Encapsulates state critic and queries them for values according to the provided policy ID."""
import dataclasses
from abc import ABC, abstractmethod
from typing import Union, List, Dict

import torch

from maze.core.agent.torch_policy import PolicyOutput, PolicySubStepOutput
from maze.core.env.observation_conversion import ObservationType, TorchObservationType
from maze.core.env.structured_env import ActorID
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord


@dataclasses.dataclass
class CriticStepOutput:
    """Critic Step output holds the output of an a critic for an individual env step."""

    values: torch.Tensor
    """The computed values."""

    detached_values: torch.Tensor
    """The computed detached values."""

    actor_id: ActorID
    """The actor id of the actor corresponding to this critic output."""


class CriticOutput:
    """Critic output holds the output of a critic for one full flat env step.

    Individual CriticStepOutputs are stored in a list which can be (if needed) referenced to the corresponding ActorID.
    """

    def __init__(self):
        self._step_critic_outputs: List[CriticStepOutput] = list()

    def append(self, value: CriticStepOutput) -> None:
        """ Set self[key] to value. """
        self._step_critic_outputs.append(value)

    @property
    def actor_ids(self) -> List[ActorID]:
        """List of actor IDs for the individual sub-steps."""
        return [cso.actor_id for cso in self._step_critic_outputs]

    @property
    def values(self) -> List[torch.Tensor]:
        """List of values for the individual sub-steps"""
        return [cso.values for cso in self._step_critic_outputs]

    @property
    def detached_values(self) -> List[torch.Tensor]:
        """List of detached values for the individual sub-steps"""
        return [cso.detached_values for cso in self._step_critic_outputs]

    def reshape(self, shape: torch.Size) -> None:
        """Reshape all the elements of the critic output to the given shape"""
        for cso in self._step_critic_outputs:
            cso.values = cso.values.reshape(shape)
            cso.detached_values = cso.detached_values.reshape(shape)


@dataclasses.dataclass
class CriticStepInput:
    """Critic input for a single substep of the env, holding the logits and the actor_ids corresponding to where the
        embedding logits where retrieved if applicable, otherwise just the corresponding actor."""

    logits: Dict[str, torch.Tensor]
    """The logits to use as an input for the critic."""

    actor_id: ActorID
    """The actor id of the corresponding actor."""


class CriticInput:
    """Critic output defined as it's own type, since it has to be explicitly build to be compatible with shared
    embedding networks."""

    def __init__(self):
        self._step_critic_inputs = list()

    def append(self, item: CriticStepInput) -> None:
        """Append an CriticStepInput object to the Critic input internal list.

        :param item: The item to add.
        """
        self._step_critic_inputs.append(item)

    def __getitem__(self, idx: int) -> CriticStepInput:
        """Retrieve an CriticStepInput item by index.

        :param idx: The index used to retrieve the element in the internal list.
        :return: The CriticStepInput at the specified place in the internal list.
        """
        return self._step_critic_inputs[idx]

    @property
    def logits(self) -> List[Dict[str, torch.Tensor]]:
        """List of logits for the individual sub-steps."""
        return [csi.logits for csi in self._step_critic_inputs]

    @property
    def actor_ids(self) -> List[ActorID]:
        """List of actor IDs for the individual sub-steps."""
        return [cso.actor_id for cso in self._step_critic_inputs]

    @property
    def substep_inputs(self) -> List[CriticStepInput]:
        """List of CriticStepInputs for the individual sub-steps."""
        return self._step_critic_inputs


class StateCritic(ABC):
    """Structured state critic class designed to work with structured environments.
    (see :class:`~maze.core.env.structured_env.StructuredEnv`).

    It encapsulates state critic and queries them for values according to the provided policy ID.
    """

    @abstractmethod
    def predict_values(self, critic_input: CriticInput) -> CriticOutput:
        """Query a critic that corresponds to the given ID for the state value.

        :param critic_input: The critic input for predicting the values of all sub-steps of the env.
        :return: Critic output holding the values, detached values and actor_id
        """

    @abstractmethod
    def predict_value(self, observation: ObservationType, critic_id: Union[int, str]) -> CriticStepOutput:
        """Query a critic that corresponds to the given ID for the state value.

        :param observation: Current observation of the environment
        :param critic_id: The critic id to query
        :return: The value for this observation
        """

    @staticmethod
    def build_step_critic_input(policy_step_output: PolicySubStepOutput, observation: TorchObservationType) \
            -> CriticStepInput:
        """Build the critic input for an individual step, by combining the policy step output and the given observation.

        :param policy_step_output: The output of the corresponding policy ot check for shared embedding outputs.
        :param observation: The observation as the default input to the critic.

        :return: The Critic input for this specific step.
        """
        if policy_step_output.embedding_logits is not None:
            combined_keys = list(policy_step_output.embedding_logits.keys()) + list(observation.keys())
            assert len(set(combined_keys)) == len(combined_keys), \
                f'Duplicates in critic input found, please make sure that no shared embedding keys/outputs have the ' \
                f'same name as any observation.'
            combined_logits = policy_step_output.embedding_logits
            combined_logits.update(observation)
            return CriticStepInput(logits=combined_logits,
                                   actor_id=policy_step_output.actor_id)
        else:
            return CriticStepInput(logits=observation, actor_id=policy_step_output.actor_id)

    @staticmethod
    def build_critic_input(policy_output: PolicyOutput, record: StructuredSpacesRecord) -> CriticInput:
        """Build the critic input from the policy outputs and the spaces record (policy input).

        This method is responsible for building a List that hold the appropriate input for each critic w.r.t. the
        substep and the shared-embedding-keys.

        :param policy_output: The full policy output.
        :param record: The structured spaces record used to compute the policy output.
        :return: A Critic input.
        """
        critic_input = CriticInput()
        for idx, substep_record in enumerate(record.substep_records):
            assert substep_record.actor_id == policy_output[idx].actor_id
            critic_input.append(StateCritic.build_step_critic_input(policy_output[idx], substep_record.observation))

        return critic_input
