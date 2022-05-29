""" Implements an action recording wrapper. """
from typing import Dict, Any, Tuple, Optional, Union

from pathlib import Path

from maze.core.annotations import override
from maze.core.env.core_env import CoreEnv
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.trajectory_recording.records.action_record import ActionRecord
from maze.core.wrappers.wrapper import ObservationWrapper, Wrapper


class ActionRecordingWrapper(Wrapper[MazeEnv]):
    """An Action Recording Wrapper that records for (sub-)step the respective MazeAction or agent action taken.

    :param env: Environment to wrap.
    :param record_maze_actions: If True maze action objects are recorded.
    :param record_actions: If True agent actions are recorded.
    :param output_dir: Path where to store the action records.
    """

    def __init__(self, env: MazeEnv, record_maze_actions: bool, record_actions: bool,
                 output_dir: str = 'action_records'):
        super().__init__(env)
        self.record_maze_actions = record_maze_actions
        self.record_actions = record_actions

        self.action_record = None
        self._episode_id = None
        self._current_seed = None
        self._cum_reward = None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @override(ObservationWrapper)
    def step(self, action) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Intercept ``ObservationWrapper.step`` and map observation."""

        # get current actor id
        actor_id = self.env.actor_id()
        curr_env_time = self.env.get_env_time()

        # take actual step
        observation, reward, done, info = self.env.step(action)
        self._cum_reward += reward

        # record action taken
        last_action = self.env.get_maze_action()
        if self.record_maze_actions:
            self.action_record.set_maze_action(curr_env_time, maze_action=last_action)

        if self.record_actions:
            self.action_record.set_agent_action(curr_env_time, actor_id=actor_id, action=action)

        return observation, reward, done, info

    @override(CoreEnv)
    def seed(self, seed: int) -> None:
        """Sets the seed for this environment's random number generator(s).

        :param: seed: the seed integer initializing the random number generator.
        """
        self._current_seed = seed
        self.env.seed(seed)

    @override(ObservationWrapper)
    def reset(self) -> Any:
        """Intercept ``ObservationWrapper.reset`` and map observation."""

        # make sure that the episode is seeded properly
        assert self._current_seed is not None

        # dump previous trajectory
        self.dump()

        self.action_record = ActionRecord(seed=self._current_seed)
        obs = self.env.reset()
        self._cum_reward = 0.0

        self._episode_id = self.env.get_episode_id()

        # clear seed to make sure that the next episode is again seeded properly
        self._current_seed = None

        return obs

    def dump(self) -> None:
        """Dump recorded trajectory to file.
        """
        output_path = self.output_dir / f"{self._episode_id}.pkl"
        if self.action_record is not None:
            # set cumulative reward
            self.action_record.cum_action_record_reward = self._cum_reward
            # dump record
            self.action_record.dump(output_path)

    def clone_from(self, env: 'ActionRecordingWrapper') -> None:
        """Reset this gym environment to the given state by creating a deep copy of the `env.state` instance variable"""
        raise RuntimeError("Cloning the 'ActionRecordingWrapper' is not supported.")

    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType], first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        raise NotImplementedError
