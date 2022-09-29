"""Wrapper for recording raw actions and observation, as seen in a particular
point in the wrapper stack (where this wrapper is placed)."""
import pickle
from pathlib import Path
from typing import Union, Any, Tuple, Dict, Optional

import gym

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.trajectory_recording.records.spaces_record import SpacesRecord
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord
from maze.core.wrappers.wrapper import Wrapper


class SpacesRecordingWrapper(Wrapper[MazeEnv]):
    """Records actions, observations, rewards and dones in structured spaces records.

    Dumps one trajectory record file per episode.

    :param output_dir: Where to dump the serialized trajectory data. Files for individual episodes
                       will be named after the episode ID, with ".pkl" suffix.
    """

    def __init__(self, env: Union[gym.Env, MazeEnv], output_dir: str = "space_records"):
        super().__init__(env)

        self.episode_record: Optional[SpacesTrajectoryRecord] = None
        self.last_observation = Optional[ObservationType]
        self.last_env_time: Optional[int] = None

        self.output_dir = Path(output_dir)

    @override(BaseEnv)
    def reset(self) -> Any:
        """Write the episode record and initialize a new one."""
        self.write_episode_record()

        self.last_observation = self.env.reset()
        self.last_env_time = None
        self.episode_record = SpacesTrajectoryRecord(id=self.env.get_episode_id())

        return self.last_observation

    @override(BaseEnv)
    def step(self, action: ActionType) -> Tuple[ObservationType, Any, bool, Dict[Any, Any]]:
        """Record available step-level data."""
        assert self.episode_record is not None, "Environment must be reset before stepping."

        # If the env time changed, start a new structured step record
        if self.env.get_env_time() != self.last_env_time:
            self.episode_record.step_records.append(StructuredSpacesRecord())
            self.last_env_time = self.env.get_env_time()

        actor_id = self.env.actor_id()  # Get actor Id before the step, so it corresponds to the action taken
        observation, reward, done, info = self.env.step(action)

        # Record the spaces of the current (sub)step
        self.episode_record.step_records[-1].append(SpacesRecord(
            actor_id=actor_id,
            observation=self.last_observation,
            action=action,
            reward=reward,
            done=done,
            info=info
        ))

        self.last_observation = observation
        return observation, reward, done, info

    def write_episode_record(self) -> None:
        """Serializes the episode record, if available."""
        if self.episode_record and len(self.episode_record.step_records) > 0:
            output_path = self.output_dir / f"{self.episode_record.id}.pkl"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as out_f:
                pickle.dump(self.episode_record, out_f)

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Keep both actions and observation the same - no change takes place in this wrapper."""
        return self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)
