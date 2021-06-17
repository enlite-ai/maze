"""Encapsulates state critic and queries them for values according to the provided policy ID."""
from abc import ABC, abstractmethod
from typing import Union

from maze.core.agent.state_critic_input_output import CriticStepOutput, CriticOutput, CriticStepInput, CriticInput
from maze.core.agent.torch_policy_output import PolicySubStepOutput, PolicyOutput
from maze.core.env.observation_conversion import ObservationType, TorchObservationType
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord


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
            return CriticStepInput(tensor_dict=combined_logits,
                                   actor_id=policy_step_output.actor_id)
        else:
            return CriticStepInput(tensor_dict=observation, actor_id=policy_step_output.actor_id)

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
