"""File holding all the state critic input and output classes."""
import dataclasses
from typing import List, Sequence, Dict

import torch

from maze.core.env.structured_env import ActorID


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

    def reshape(self, shape: Sequence[int]) -> None:
        """Reshape all the elements of the critic output to the given shape"""
        for cso in self._step_critic_outputs:
            cso.values = cso.values.reshape(shape)
            cso.detached_values = cso.detached_values.reshape(shape)


@dataclasses.dataclass
class CriticStepInput:
    """Critic input for a single substep of the env, holding the tensor_dict and the actor_ids corresponding to where
        the embedding logits where retrieved if applicable, otherwise just the corresponding actor.

        The tensor dict here holds the observations of the corresponding env-sub-step as well as logits coming from the
        actor if a shared embedding is used.
        """

    tensor_dict: Dict[str, torch.Tensor]
    """The tensor dict to use as an input for the critic."""

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
    def tensor_dict(self) -> List[Dict[str, torch.Tensor]]:
        """List of tensor dicts for the individual sub-steps."""
        return [csi.tensor_dict for csi in self._step_critic_inputs]

    @property
    def actor_ids(self) -> List[ActorID]:
        """List of actor IDs for the individual sub-steps."""
        return [cso.actor_id for cso in self._step_critic_inputs]

    @property
    def substep_inputs(self) -> List[CriticStepInput]:
        """List of CriticStepInputs for the individual sub-steps."""
        return self._step_critic_inputs
