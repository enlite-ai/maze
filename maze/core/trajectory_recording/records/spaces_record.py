"""Record of spaces (i.e., raw action, observation, and associated data) from a single sub-step."""

from dataclasses import dataclass
from typing import Dict, Union, Optional, List

import numpy as np
import torch

from maze.core.env.structured_env import ActorID
from maze.perception.perception_utils import convert_to_numpy, convert_to_torch
from maze.train.utils.train_utils import stack_numpy_dict_list, stack_torch_dict_list


@dataclass
class SpacesRecord:
    """Record of spaces (i.e., raw action, observation, and associated data) from a single sub-step."""

    actor_id: ActorID
    """ID of the actor for this step."""

    observation: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None
    """Observation recorded during the step."""

    action: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None
    """Action recorded during the step."""

    reward: Optional[Union[float, np.ndarray, torch.Tensor]] = None
    """Reward recorded during the step."""

    done: Optional[Union[bool, np.ndarray, torch.Tensor]] = None
    """Done flag recorded during the step."""

    info: Optional[Dict] = None
    """Info dictionary recorded during the step."""

    next_observation: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None
    """Observation obtained after this step (i.e., results of the action taken in this step)."""

    logits: Optional[Dict[str, np.ndarray]] = None
    """Action logits recorded during the step."""

    discounted_return: Optional[Union[float, np.ndarray]] = None
    """Discounted return for this step."""

    batch_shape: Optional[List[int]] = None
    """If the record is batched, this is the shape of the batch."""

    @classmethod
    def stack(cls, records: List['SpacesRecord']) -> 'SpacesRecord':
        """Stack multiple records into a single spaces record. Useful for processing multiple records in a batch.

        All the records should be in numpy and have the same structure of the spaces (i.e. come from the same
        environment etc.).

        :param records: Records to stack.
        :return: Single stacked record, containing all the given records, and having the corresponding batch shape.
        """

        assert len(set([r.substep_key for r in records])) == 1, "Cannot batch records for different sub-step keys."

        stacked_record = SpacesRecord(
            actor_id=records[0].actor_id,
            observation=stack_numpy_dict_list([r.observation for r in records]),
            action=stack_numpy_dict_list([r.action for r in records]),
            reward=np.stack([r.reward for r in records]),
            done=np.stack([r.done for r in records])
        )

        if records[0].next_observation:
            stacked_record.next_observation = stack_numpy_dict_list([r.next_observation for r in records])

        if records[0].logits:
            stacked_record.logits = stack_torch_dict_list([r.logits for r in records])

        stacked_record.batch_shape = [len(records)]
        if records[0].batch_shape:
            stacked_record.batch_shape += records[0].batch_shape

        return stacked_record

    @property
    def substep_key(self) -> Union[str, int]:
        """Sub-step key (i.e., the first part of the Actor ID) for this step."""
        return self.actor_id.step_key

    @property
    def agent_id(self) -> int:
        """Sub-step key (i.e., the second part of the Actor ID) for this step."""
        return self.actor_id.agent_id

    def to_numpy(self) -> 'SpacesRecord':
        """Convert the record to numpy."""
        self.observation = convert_to_numpy(self.observation, cast=None, in_place=True)
        self.action = convert_to_numpy(self.action, cast=None, in_place=True)
        self.reward = self.reward.cpu().numpy()
        self.done = self.done.cpu().numpy()

        if self.next_observation is not None:
            self.next_observation = convert_to_numpy(self.next_observation, cast=None, in_place=True)

        if self.logits is not None:
            self.logits = convert_to_numpy(self.logits, cast=None, in_place=True)

        return self

    def to_torch(self, device: str) -> 'SpacesRecord':
        """Convert the record to Torch.

        :param device: Device to move the tensors to.
        """
        self.observation = convert_to_torch(self.observation, device=device, cast=None, in_place=True)
        self.action = convert_to_torch(self.action, device=device, cast=None, in_place=True)
        self.reward = torch.from_numpy(np.asarray(self.reward)).to(device)
        self.done = torch.from_numpy(np.asarray(self.done)).to(device)

        if self.next_observation is not None:
            self.next_observation = convert_to_torch(self.next_observation, device=device, cast=None, in_place=True)

        if self.logits is not None:
            self.logits = convert_to_torch(self.logits, device=device, cast=None, in_place=True)

        return self

    def __repr__(self):
        return f"Spaces record (batch_shape={self.batch_shape}): Actor {self.actor_id}, " \
               f"observation keys {list(self.observation.keys())}, " \
               f"action keys {list(self.action.keys())}"
