"""Record of spaces (i.e., raw action, observation, and associated data) from a single sub-step."""

from __future__ import annotations

from dataclasses import dataclass

from maze.core.env.structured_env import ActorID
from maze.perception.perception_utils import convert_to_numpy, convert_to_torch
from maze.train.utils.train_utils import stack_numpy_dict_list, stack_torch_dict_list

import numpy as np
import torch

PolicyRecordType = object


@dataclass
class SpacesRecord:
    """Record of spaces (i.e., raw action, observation, and associated data) from a single sub-step."""

    actor_id: ActorID
    """ID of the actor for this step."""

    observation: dict[str, np.ndarray | torch.Tensor] | None = None
    """Observation recorded during the step."""

    action: dict[str, np.ndarray | torch.Tensor] | None = None
    """Action recorded during the step."""

    reward: float | np.ndarray | torch.Tensor | None = None
    """Reward recorded during the step."""

    terminated: bool | np.ndarray | torch.Tensor | None = None
    """Terminated flag recorded during the step."""

    truncated: bool | np.ndarray | torch.Tensor | None = None
    """Truncated flag recorded during the step."""

    info: dict | None = None
    """Info dictionary recorded during the step."""

    next_observation: dict[str, np.ndarray | torch.Tensor] | None = None
    """Observation obtained after this step (i.e., results of the action taken in this step)."""

    logits: dict[str, np.ndarray] | None = None
    """Action logits recorded during the step."""

    discounted_return: float | np.ndarray | None = None
    """Discounted return for this step."""

    batch_shape: list[int] | None = None
    """If the record is batched, this is the shape of the batch."""

    policy_record: PolicyRecordType | None = None
    """Policy specific data that can be recorded with the help of the write_policy_record method of the policy."""

    env_time: int | None = None
    """The env time (t) of the env when recording the observation such that:
       (s_t, a_t, v_t) -> env step -> (r_t, terminated_t, truncated_t, info_t) is recorded.
    """

    @classmethod
    def stack(cls, records: list[SpacesRecord]) -> SpacesRecord:
        """Stack multiple records into a single spaces record. Useful for processing multiple records in a batch.

        All the records should be in numpy and have the same structure of the spaces (i.e. come from the same
        environment etc.).

        :param records: Records to stack.
        :return: Single stacked record, containing all the given records, and having the corresponding batch shape.
        """

        assert len({r.substep_key for r in records}) == 1, 'Cannot batch records for different sub-step keys.'
        assert len({r.agent_id for r in records}) == 1, 'Cannot batch records for different agent ids.'

        stacked_record = SpacesRecord(
            actor_id=records[0].actor_id,
            observation=stack_numpy_dict_list([r.observation for r in records]),
            action=stack_numpy_dict_list([r.action for r in records]),
            reward=np.stack([r.reward for r in records]),
            terminated=np.stack([r.terminated for r in records]),
            truncated=np.stack([r.truncated for r in records]),
        )

        if records[0].next_observation:
            stacked_record.next_observation = stack_numpy_dict_list([r.next_observation for r in records])

        if records[0].logits:
            stacked_record.logits = stack_torch_dict_list([r.logits for r in records])

        if records[0].policy_record:
            raise NotImplementedError('Stacking is not implemented for policy records')

        stacked_record.batch_shape = [len(records)]
        if records[0].batch_shape:
            stacked_record.batch_shape += records[0].batch_shape

        return stacked_record

    @property
    def substep_key(self) -> str | int:
        """Sub-step key (i.e., the first part of the Actor ID) for this step."""
        return self.actor_id.step_key

    @property
    def agent_id(self) -> int:
        """Sub-step key (i.e., the second part of the Actor ID) for this step."""
        return self.actor_id.agent_id

    @property
    def done(self) -> bool:
        """Whether the step is done (i.e., terminated or truncated)."""
        return self.terminated | self.truncated

    def to_numpy(self) -> SpacesRecord:
        """Convert the record to numpy."""
        self.observation = convert_to_numpy(self.observation, cast=None, in_place=True)
        self.action = convert_to_numpy(self.action, cast=None, in_place=True)
        self.reward = self.reward.cpu().numpy()
        self.terminated = self.terminated.cpu().numpy()
        self.truncated = self.truncated.cpu().numpy()

        if self.next_observation is not None:
            self.next_observation = convert_to_numpy(self.next_observation, cast=None, in_place=True)

        if self.logits is not None:
            self.logits = convert_to_numpy(self.logits, cast=None, in_place=True)

        return self

    def to_torch(self, device: str) -> SpacesRecord:
        """Convert the record to Torch.

        :param device: Device to move the tensors to.
        """
        self.observation = convert_to_torch(self.observation, device=device, cast=None, in_place=True)
        self.action = convert_to_torch(self.action, device=device, cast=None, in_place=True)
        self.reward = torch.from_numpy(np.asarray(self.reward)).to(device)
        self.terminated = torch.from_numpy(np.asarray(self.terminated)).to(device)
        self.truncated = torch.from_numpy(np.asarray(self.truncated)).to(device)

        if self.next_observation is not None:
            self.next_observation = convert_to_torch(self.next_observation, device=device, cast=None, in_place=True)

        if self.logits is not None:
            self.logits = convert_to_torch(self.logits, device=device, cast=None, in_place=True)

        return self

    def __repr__(self):
        return (
            f'Spaces record (batch_shape={self.batch_shape}): Actor {self.actor_id}, '
            f'observation keys {list(self.observation.keys())}, '
            f'action keys {list(self.action.keys())}'
        )
