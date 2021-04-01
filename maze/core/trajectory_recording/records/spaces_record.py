from dataclasses import dataclass
from typing import Dict, Union, Optional, List

import numpy as np
import torch

from maze.core.env.structured_env import ActorIDType
from maze.perception.perception_utils import convert_to_numpy, convert_to_torch
from maze.train.utils.train_utils import stack_numpy_dict_list, stack_torch_dict_list


@dataclass
class SpacesRecord:
    actor_id: ActorIDType

    observation: Optional[Dict[str, Union[torch.Tensor]]] = None
    """Dictionary of observations recorded during the step."""

    action: Optional[Dict[str, Union[torch.Tensor]]] = None
    """Dictionary of actions recorded during the step."""

    reward: Optional[Union[float, np.ndarray, torch.Tensor]] = None
    """Dictionary of rewards recorded during the step."""

    done: Optional[Union[bool, np.ndarray, torch.Tensor]] = None
    """Dictionary of dones recorded during the step."""

    info: Optional[Dict] = None
    """Dictionary of info dictionaries recorded during the step."""

    logits: Optional[Dict[str, np.ndarray]] = None
    """Dictionary of dones recorded during the step."""

    batch_shape: Optional[List[int]] = None
    """If the record is batched, this is the shape of the batch."""

    @classmethod
    def stack(cls, records: List['SpacesRecord']) -> 'SpacesRecord':
        assert len(set([r.substep_key for r in records])) == 1, "Cannot batch records for different sub-step keys."

        stacked_record = SpacesRecord(
            actor_id=records[0].actor_id,
            observation=stack_numpy_dict_list([r.observation for r in records]),
            action=stack_numpy_dict_list([r.action for r in records]),
            reward=np.stack([r.reward for r in records]),
            done=np.stack([r.done for r in records])
        )

        if records[0].logits:
            stacked_record.logits = stack_torch_dict_list([r.logits for r in records])

        stacked_record.batch_shape = [len(records)]
        if records[0].batch_shape:
            stacked_record.batch_shape += records[0].batch_shape

        return stacked_record

    @property
    def substep_key(self) -> Union[str, int]:
        return self.actor_id[0]

    @property
    def agent_id(self) -> int:
        return self.actor_id[1]

    def to_numpy(self):
        """Convert the record to numpy."""
        self.observation = convert_to_numpy(self.observation, cast=None, in_place=True)
        self.action = convert_to_numpy(self.action, cast=None, in_place=True)
        self.reward = self.reward.cpu().numpy()
        self.done = self.done.cpu().numpy()

        if self.logits is not None:
            self.logits = convert_to_numpy(self.logits, cast=None, in_place=True)

        return self

    def to_torch(self, device: str):
        """Convert the record to Torch.

        :param device: Device to move the tensors to.
        """
        self.observation = convert_to_torch(self.observation, device=device, cast=None, in_place=True)
        self.action = convert_to_torch(self.action, device=device, cast=None, in_place=True)
        self.reward = torch.from_numpy(np.asarray(self.reward)).to(device)
        self.done = torch.from_numpy(np.asarray(self.done)).to(device)

        if self.logits is not None:
            self.logits = convert_to_torch(self.logits, device=device, cast=None, in_place=True)

        return self

    def __repr__(self):
        return f"Spaces record (batch_shape={self.batch_shape}): Actor {self.actor_id}, " \
               f"observation keys {list(self.observation.keys())}, " \
               f"action keys {list(self.action.keys())}"
