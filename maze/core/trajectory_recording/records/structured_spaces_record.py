from dataclasses import dataclass
from typing import Dict, Optional, TypeVar, List, Union

import numpy as np

from maze.core.env.maze_env import MazeEnv
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.log_stats.log_stats import LogStats
from maze.core.trajectory_recording.records.state_record import StateRecord
from maze.core.trajectory_recording.records.raw_maze_state import RawState, RawMazeAction
from maze.perception.perception_utils import convert_to_numpy, convert_to_torch
from maze.train.utils.train_utils import stack_numpy_dict_list

StepKeyType = Union[str, int]


@dataclass
class StructuredSpacesRecord:
    """Records spaces (i.e., raw actions and observations) from a single environment step.

    Individual items are structured in dictionaries, with keys corresponding to the structured sub-step IDs.

    Provides helpers methods for batching, i.e. stacking of multiple spaces records to be processed by a model
    in a single batch.
    """

    observations: Dict[StepKeyType, Dict[str, np.ndarray]]
    """Dictionary of observations recorded during the step."""

    actions: Dict[StepKeyType, Dict[str, np.ndarray]]
    """Dictionary of actions recorded during the step."""

    rewards: Optional[Dict[StepKeyType, Union[float, np.ndarray]]]
    """Dictionary of rewards recorded during the step."""

    dones: Optional[Dict[StepKeyType, Union[float, bool]]]
    """Dictionary of dones recorded during the step."""

    infos: Optional[Dict[StepKeyType, Dict]] = None
    """Dictionary of info dictionaries recorded during the step."""

    logits: Optional[Dict[StepKeyType, Dict[str, np.ndarray]]] = None
    """Dictionary of dones recorded during the step."""

    event_log: Optional[StepEventLog] = None
    """Log of events recorded during the whole step."""

    step_stats: Optional[LogStats] = None
    """Statistics recorded during the whole step."""

    episode_stats: Optional[LogStats] = None
    """Aggregated statistics from the last episode. Expected to be attached only to terminal steps of episodes."""

    batch_shape: Optional[List[int]] = None
    """If the record is batched, this is the shape of the batch."""

    @classmethod
    def stack_records(cls, records: List['StructuredSpacesRecord']) -> StateRecord:
        """Stack multiple records into a single spaces record. Useful for processing multiple records in a batch.

        All the records should be in numpy and have the same structure of the spaces (i.e. come from the same
        environment etc.).

        :param records: Records to stack.
        :return: Single stacked record, containing all the given records, and having the corresponding batch shape.
        """
        logits_present = records[0].logits is not None

        stacked_record = StructuredSpacesRecord(
            observations={}, actions={}, logits={} if logits_present else None,
            rewards=stack_numpy_dict_list([r.rewards for r in records]),
            dones=stack_numpy_dict_list([r.dones for r in records]))

        # Actions and observations are nested dict spaces => need to go one level down with stacking
        for step_key in records[0].observations.keys():
            stacked_record.actions[step_key] = stack_numpy_dict_list([r.actions[step_key] for r in records])
            stacked_record.observations[step_key] = stack_numpy_dict_list([r.observations[step_key] for r in records])

            if logits_present:
                stacked_record.logits[step_key] = stack_numpy_dict_list([r.logits[step_key] for r in records])

        stacked_record.batch_shape = [len(records)] + records[0].batch_shape if records[0].batch_shape \
            else [len(records)]

        return stacked_record

    def is_batched(self):
        """Return whether this record is batched or not."""
        return self.batch_shape is not None

    def is_done(self) -> bool:
        """Return true if the episode ended during this structured step."""
        return np.any(list(self.dones.values()))

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
        return StructuredSpacesRecord(observations=obs, actions=action, rewards=None, dones=None)

    def to_numpy(self):
        """Convert the record to numpy."""
        self.observations = convert_to_numpy(self.observations, cast=None, in_place=True)
        self.actions = convert_to_numpy(self.actions, cast=None, in_place=True)
        self.rewards = convert_to_numpy(self.rewards, cast=None, in_place=True)
        self.dones = convert_to_numpy(self.dones, cast=None, in_place=True)

        if self.logits is not None:
            self.logits = convert_to_numpy(self.logits, cast=None, in_place=True)

        return self

    def to_torch(self, device: str):
        """Convert the record to Torch.

        :param device: Device to move the tensors to.
        """
        self.observations = convert_to_torch(self.observations, device=device, cast=None, in_place=True)
        self.actions = convert_to_torch(self.actions, device=device, cast=None, in_place=True)
        self.rewards = convert_to_torch(self.rewards, device=device, cast=None, in_place=True)
        self.dones = convert_to_torch(self.dones, device=device, cast=None, in_place=True)

        if self.logits is not None:
            self.logits = convert_to_torch(self.logits, device=device, cast=None, in_place=True)

        return self
