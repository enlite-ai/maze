"""Recording spaces (i.e., raw actions and observations) from a single environment step."""

from dataclasses import dataclass
from typing import Optional, List, Union, Dict

import numpy as np

from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.log_stats.log_stats import LogStats
from maze.core.trajectory_recording.records.raw_maze_state import RawState, RawMazeAction
from maze.core.trajectory_recording.records.spaces_record import SpacesRecord
from maze.core.trajectory_recording.records.state_record import StateRecord

StepKeyType = Union[str, int]


@dataclass
class StructuredSpacesRecord:
    """Records spaces (i.e., raw actions and observations) from a single environment step.

    Individual items are structured in dictionaries, with keys corresponding to the structured sub-step IDs.

    Provides helpers methods for batching, i.e. stacking of multiple spaces records to be processed by a model
    in a single batch.
    """

    substep_records: List[SpacesRecord] = None
    """Records for individual sub-steps (containing individual observations, action etc.)"""

    event_log: Optional[StepEventLog] = None
    """Log of events recorded during the whole step."""

    step_stats: Optional[LogStats] = None
    """Statistics recorded during the whole step."""

    episode_stats: Optional[LogStats] = None
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
    def stack_records(cls, records: List['StructuredSpacesRecord']) -> 'StructuredSpacesRecord':
        """Stack multiple records into a single spaces record. Useful for processing multiple records in a batch.

        All the records should be in numpy and have the same structure of the spaces (i.e. come from the same
        environment etc.).

        :param records: Records to stack.
        :return: Single stacked record, containing all the given records, and having the corresponding batch shape.
        """
        stacked_substeps = []

        for substep_records in zip(*[r.substep_records for r in records]):
            stacked_substeps.append(SpacesRecord.stack(substep_records))

        return StructuredSpacesRecord(substep_records=stacked_substeps)

    @classmethod
    def converted_from(cls, state_record: StateRecord, conversion_env: MazeEnv, first_step_in_episode: bool) \
            -> 'StructuredSpacesRecord':
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
        obs = state_record.maze_state.observation if isinstance(state_record.maze_state,
                                                                RawState) else state_record.maze_state
        action = state_record.maze_action.action if isinstance(state_record.maze_action,
                                                               RawMazeAction) else state_record.maze_action

        obs, action = conversion_env.get_observation_and_action_dicts(obs, action, first_step_in_episode)

        substep_records = [SpacesRecord(
            actor_id=ActorID(substep_key, 0),
            observation=obs[substep_key],
            action=action[substep_key],
            reward=None,
            done=None
        ) for substep_key in obs.keys()]

        return StructuredSpacesRecord(substep_records=substep_records)

    def to_numpy(self) -> 'StructuredSpacesRecord':
        """Convert the record to numpy."""
        for substep_record in self.substep_records:
            substep_record.to_numpy()
        return self

    def to_torch(self, device: str) -> 'StructuredSpacesRecord':
        """Convert the record to Torch.

        :param device: Device to move the tensors to.
        :return: Self after conversion.
        """
        for substep_record in self.substep_records:
            substep_record.to_torch(device=device)
        return self

    def __repr__(self) -> str:
        repr_str = "Structured spaces record:"
        for substep_record in self.substep_records:
            repr_str += f"\n - {substep_record}"
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

    def is_done(self) -> bool:
        """Return true if the episode ended during this structured step.

        :return: true if the episode ended during this structured step
        """
        return self.substep_records[-1].done

    @property
    def actor_ids(self) -> List[ActorID]:
        """List of actor IDs for the individual sub-steps."""
        return [r.actor_id for r in self.substep_records]

    @property
    def substep_keys(self) -> List[StepKeyType]:
        """List of sub-step keys for the individual sub-steps."""
        return [r.substep_key for r in self.substep_records]

    @property
    def actions(self) -> List[ActionType]:
        """List of actions from the individual sub-steps."""
        return [r.action for r in self.substep_records]

    @property
    def observations(self) -> List[ObservationType]:
        """List of observations from the individual sub-steps."""
        return [r.observation for r in self.substep_records]

    @property
    def rewards(self) -> List[Union[float, np.ndarray]]:
        """List of rewards from the individual sub-steps."""
        return [r.reward for r in self.substep_records]

    @property
    def dones(self) -> List[bool]:
        """List of dones from the individual sub-steps."""
        return [r.done for r in self.substep_records]

    @property
    def next_observations(self) -> List[ObservationType]:
        """List of next observations from the individual sub-steps."""
        return [r.next_observation for r in self.substep_records]

    @property
    def logits(self) -> List[Dict[str, np.ndarray]]:
        """List of logits from the individual sub-steps."""
        return [r.logits for r in self.substep_records]

    @property
    def discounted_returns(self) -> List[Union[float, np.ndarray]]:
        """List of discounted returns from the individual sub-steps."""
        return [r.discounted_return for r in self.substep_records]

    @property
    def actions_dict(self) -> Dict[StepKeyType, ActionType]:
        """Dict of actions from the sub-steps, keyed by the sub-step ID (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.action for r in self.substep_records}

    @property
    def observations_dict(self) -> Dict[StepKeyType, ObservationType]:
        """Dict of observations from the sub-steps, keyed by the sub-step ID (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.observation for r in self.substep_records}

    @property
    def rewards_dict(self) -> Dict[StepKeyType, Union[float, np.ndarray]]:
        """Dict of rewards from the sub-steps, keyed by the sub-step ID (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.reward for r in self.substep_records}

    @property
    def dones_dict(self) -> Dict[StepKeyType, bool]:
        """Dict of dones from the sub-steps, keyed by the sub-step ID (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.done for r in self.substep_records}

    @property
    def next_observations_dict(self) -> Dict[StepKeyType, ObservationType]:
        """Dict of next observations from the sub-steps, keyed by the sub-step ID
        (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.next_observation for r in self.substep_records}

    @property
    def logits_dict(self) -> Dict[StepKeyType, Dict[str, np.ndarray]]:
        """Dict of logits from the sub-steps, keyed by the sub-step ID (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.logits for r in self.substep_records}

    @property
    def discounted_returns_dict(self) -> Dict[StepKeyType, Union[float, np.ndarray]]:
        """Dict of discounted returns from the sub-steps, keyed by the sub-step ID
        (not suitable in multi-agent scenarios)."""
        return {r.substep_key: r.discounted_return for r in self.substep_records}
