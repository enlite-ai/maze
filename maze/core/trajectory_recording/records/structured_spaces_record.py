"""Recording spaces (i.e., raw actions and observations) from a single environment step."""

from __future__ import annotations

from dataclasses import dataclass

from maze.core.env.action_conversion import ActionType, TorchActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType, TorchObservationType
from maze.core.env.structured_env import ActorID
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.log_stats.log_stats import LogStats
from maze.core.trajectory_recording.records.raw_maze_state import (
    RawMazeAction,
    RawState,
)
from maze.core.trajectory_recording.records.spaces_record import (
    PolicyRecordType,
    SpacesRecord,
)
from maze.core.trajectory_recording.records.state_record import StateRecord

import numpy as np
import torch

StepKeyType = str | int


@dataclass
class StructuredSpacesRecord:
    """Records spaces (i.e., raw actions and observations) from a single environment step.

    Individual items are structured in dictionaries, with keys corresponding to the structured sub-step IDs.

    Provides helpers methods for batching, i.e. stacking of multiple spaces records to be processed by a model
    in a single batch.
    """

    substep_records: list[SpacesRecord] = None
    """Records for individual sub-steps (containing individual observations, action etc.)"""

    event_log: StepEventLog | None = None
    """Log of events recorded during the whole step."""

    step_stats: LogStats | None = None
    """Statistics recorded during the whole step."""

    episode_stats: LogStats | None = None
    """Aggregated statistics from the last episode. Expected to be attached only to terminal steps of episodes."""

    def __post_init__(self):
        if self.substep_records is None:
            self.substep_records = []

    def append(self, substep_record: SpacesRecord) -> None:
        """Append a sub-step record."""
        self.substep_records.append(substep_record)

    def __len__(self) -> int:
        """Return count of sub-step records."""
        return len(self.substep_records)

    @classmethod
    def stack_records(cls, records: list[StructuredSpacesRecord]) -> StructuredSpacesRecord:
        """Stack multiple records into a single spaces record. Useful for processing multiple records in a batch.

        All the records should be in numpy and have the same structure of the spaces (i.e. come from the same
        environment etc.).

        :param records: Records to stack.
        :return: Single stacked record, containing all the given records, and having the corresponding batch shape.
        """
        stacked_substeps = []

        assert len({len(r) for r in records}) == 1, f'records are not all of equal length: {[len(r) for r in records]}'

        for substep_records in zip(*[r.substep_records for r in records], strict=False):
            stacked_substeps.append(SpacesRecord.stack(substep_records))

        return StructuredSpacesRecord(substep_records=stacked_substeps)

    @classmethod
    def converted_from(
        cls,
        state_record: StateRecord,
        conversion_env: MazeEnv,
        first_step_in_episode: bool,
    ) -> StructuredSpacesRecord:
        """Convert a state record (containing a Maze state and Maze action) into a spaces record (containing
        raw actions and observations for each sub-step).

        Maze states and actions are converted to spaces using the supplied conversion env -- it's action and
        observation interfaces, as well as the wrapper stack determine the format of the converted actions
        and observations.

        This is useful e.g. for behavioral cloning, when we have recorded Maze states and actions from teacher runs,
        and now need to convert these into raw actions and observations to be fed to a model.

        Note that multi-agent scenarios are not supported yet (the conversion only support a single
        action-observation pair per sub-step key).

        :param state_record: State record to convert.
        :param conversion_env: Environment to use for the conversion. Determines the format of the resulting spaces.
        :param first_step_in_episode: Flag whether this is the first step in an episode (to resets stateful wrapper)
        :return: Converted spaces record.
        """
        obs = (
            state_record.maze_state.observation
            if isinstance(state_record.maze_state, RawState)
            else state_record.maze_state
        )
        action = (
            state_record.maze_action.action
            if isinstance(state_record.maze_action, RawMazeAction)
            else state_record.maze_action
        )

        obs, action = conversion_env.get_observation_and_action_dicts(obs, action, first_step_in_episode)

        substep_records = [
            SpacesRecord(
                actor_id=ActorID(substep_key, 0),
                observation=obs[substep_key],
                action=action[substep_key],
                reward=None,
                terminated=None,
                truncated=None,
            )
            for substep_key in obs.keys()
        ]

        substep_records[-1].terminated = state_record.terminated
        substep_records[-1].truncated = state_record.truncated
        substep_records[-1].reward = state_record.reward

        return StructuredSpacesRecord(substep_records=substep_records)

    def to_numpy(self) -> StructuredSpacesRecord:
        """Convert the record to numpy."""
        for substep_record in self.substep_records:
            substep_record.to_numpy()
        return self

    def to_torch(self, device: str) -> StructuredSpacesRecord:
        """Convert the record to Torch.

        :param device: Device to move the tensors to.
        :return: Self after conversion.
        """
        for substep_record in self.substep_records:
            substep_record.to_torch(device=device)
        return self

    def __repr__(self) -> str:
        repr_str = 'Structured spaces record:'
        for substep_record in self.substep_records:
            repr_str += f'\n - {substep_record}'
        return repr_str

    # -- Convenience accessors --

    def is_batched(self) -> bool:
        """Return whether this record is batched or not.

        :return: whether this record is batched or not
        """
        return self.substep_records[0].batch_shape is not None

    @property
    def batch_shape(self):
        """Return whether this record is batched or not."""
        return self.substep_records[0].batch_shape

    def is_terminated(self) -> bool:
        """Return true if the episode ended during this structured step."""
        assert not self.is_batched(), 'cannot determine done state for batched trajectory.'
        return self.substep_records[-1].terminated

    def is_truncated(self) -> bool:
        """Return true if the episode was truncated during this structured step."""
        assert not self.is_batched(), 'cannot determine done state for batched trajectory.'
        return self.substep_records[-1].truncated

    def is_done(self):
        """Return true if the episode ended during this structured step, either terminated or truncated"""
        assert not self.is_batched(), 'cannot determine done state for batched trajectory.'
        return self.substep_records[-1].truncated or self.substep_records[-1].terminated

    @property
    def actor_ids(self) -> list[ActorID]:
        """List of actor IDs for the individual sub-steps."""
        return [r.actor_id for r in self.substep_records]

    @property
    def substep_keys(self) -> list[StepKeyType]:
        """List of sub-step keys for the individual sub-steps."""
        return [r.substep_key for r in self.substep_records]

    @property
    def actions(self) -> list[ActionType | TorchActionType]:
        """List of actions from the individual sub-steps."""
        return [r.action for r in self.substep_records]

    @property
    def observations(self) -> list[ObservationType | TorchObservationType]:
        """List of observations from the individual sub-steps."""
        return [r.observation for r in self.substep_records]

    @property
    def rewards(self) -> list[float | np.ndarray | torch.Tensor]:
        """List of rewards from the individual sub-steps."""
        return [r.reward for r in self.substep_records]

    @property
    def terminated(self) -> list[bool | torch.Tensor]:
        """List of terminated flags from the individual sub-steps."""
        return [r.terminated for r in self.substep_records]

    @property
    def truncated(self) -> list[bool | torch.Tensor]:
        """List of truncated flags from the individual sub-steps."""
        return [r.truncated for r in self.substep_records]

    @property
    def next_observations(self) -> list[ObservationType | TorchObservationType]:
        """List of next observations from the individual sub-steps."""
        return [r.next_observation for r in self.substep_records]

    @property
    def logits(self) -> list[dict[str, np.ndarray | torch.Tensor]]:
        """List of logits from the individual sub-steps."""
        return [r.logits for r in self.substep_records]

    @property
    def discounted_returns(self) -> list[float | np.ndarray | torch.Tensor]:
        """List of discounted returns from the individual sub-steps."""
        return [r.discounted_return for r in self.substep_records]

    @property
    def policy_records(self) -> list[PolicyRecordType]:
        """List of policy records for the individual sub-steps."""
        return [r.policy_record for r in self.substep_records]

    @property
    def actions_dict(self) -> dict[StepKeyType, ActionType | TorchActionType]:
        """Dict of actions from the sub-steps, keyed by the sub-step ID (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.action for r in self.substep_records}

    @property
    def observations_dict(
        self,
    ) -> dict[StepKeyType, ObservationType | TorchObservationType]:
        """Dict of observations from the sub-steps, keyed by the sub-step ID (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.observation for r in self.substep_records}

    @property
    def rewards_dict(self) -> dict[StepKeyType, float | np.ndarray | torch.Tensor]:
        """Dict of rewards from the sub-steps, keyed by the sub-step ID (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.reward for r in self.substep_records}

    @property
    def terminated_dict(self) -> dict[StepKeyType, bool | torch.Tensor]:
        """Dict of terminating from the sub-steps, keyed by the sub-step ID (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.terminated for r in self.substep_records}

    @property
    def truncated_dict(self) -> dict[StepKeyType, bool | torch.Tensor]:
        """Dict of truncated flags from the sub-steps, keyed by the sub-step ID
        (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.truncated for r in self.substep_records}

    @property
    def next_observations_dict(
        self,
    ) -> dict[StepKeyType, ObservationType | TorchObservationType]:
        """Dict of next observations from the sub-steps, keyed by the sub-step ID
        (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.next_observation for r in self.substep_records}

    @property
    def logits_dict(
        self,
    ) -> dict[StepKeyType, dict[str, torch.Tensor | np.ndarray]]:
        """Dict of logits from the sub-steps, keyed by the sub-step ID (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.logits for r in self.substep_records}

    @property
    def discounted_returns_dict(
        self,
    ) -> dict[StepKeyType, float | np.ndarray | torch.Tensor]:
        """Dict of discounted returns from the sub-steps, keyed by the sub-step ID
        (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.discounted_return for r in self.substep_records}

    @property
    def policy_records_dict(self) -> dict[StepKeyType, PolicyRecordType]:
        """List of policy records for the individual sub-steps."""
        return {r.substep_key: r.policy_record for r in self.substep_records}
