"""Generate trajectory data for the wrapped environment."""

import uuid
from copy import deepcopy
from typing import Union, Any, Tuple, Dict, Optional

from maze.core.annotations import override
from maze.core.env.base_env import BaseEnv
from maze.core.env.event_env_mixin import EventEnvMixin
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.recordable_env_mixin import RecordableEnvMixin
from maze.core.env.serializable_env_mixin import SerializableEnvMixin
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.time_env_mixin import TimeEnvMixin
from maze.core.events.event_collection import EventCollection
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.rendering.keyboard_controlled_trajectory_viewer import KeyboardControlledTrajectoryViewer
from maze.core.trajectory_recording.records.raw_maze_state import RawState, RawMazeAction
from maze.core.trajectory_recording.records.state_record import StateRecord
from maze.core.trajectory_recording.records.trajectory_record import StateTrajectoryRecord
from maze.core.trajectory_recording.writers.trajectory_writer_registry import TrajectoryWriterRegistry
from maze.core.wrappers.wrapper import Wrapper


class TrajectoryRecordingWrapper(Wrapper[MazeEnv]):
    """A trajectory recording wrapper. Supports both standard gym envs and BaseEnvs, as well as environments
    that provide more access -- SerializableEnvMixin for access to environment components
    and RecordableEnvMixin for access to MazeState and MazeExecution objects.

    :param env: the environment to wrap.
    """

    def __init__(self, env: MazeEnv):
        """Avoid calling this constructor directly, use :method:`wrap` instead."""
        # BaseEnv is a subset of gym.Env
        super().__init__(env)

        self.episode_record: Optional[StateTrajectoryRecord] = None

        self.last_env_time: Optional[int] = None
        self.last_maze_state: Optional[MazeStateType] = None
        self.last_serializable_components: Optional[Dict[str, Any]] = None

        # Recording of maze actions with support for step-skipping
        self._last_maze_action_before_skipping = None  # Last maze action taken before step-skipping started (if in use)
        self._maze_action_recorded = False  # Flag used for avoiding recording of actions issued during step skipping

        # Post-step hook, to be able to record the maze action after it's been converted in MazeEnv,
        # but before step skipping starts
        self.env.context.register_post_step(self._record_last_action)

    def _record_last_action(self) -> None:
        """Record the last action unless it has been done for this step already."""
        if not self._maze_action_recorded:
            self._last_maze_action_before_skipping = deepcopy(self.env.get_maze_action())
            self._maze_action_recorded = True

    @override(BaseEnv)
    def step(self, action: Any) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Record available step-level data."""
        assert self.episode_record is not None, "Environment must be reset before stepping."

        self._maze_action_recorded = False
        observation, reward, done, info = self.env.step(action)

        # Recording of event logs and stats happens:
        #  - for TimeEnvs:   Only if the env time changed, so that we record once per time step
        #  - for other envs: Every step
        if not isinstance(self.env, TimeEnvMixin) or self.env.get_env_time() != self.last_env_time:
            # Record maze action: In case no step-skipping was done and the action has not been recorded yet, do it here
            self._record_last_action()

            # Collect step events
            event_collection = EventCollection(
                self.env.get_step_events() if isinstance(self.env, EventEnvMixin) else [])
            step_event_log = StepEventLog(self.last_env_time, events=event_collection)

            # Record trajectory data
            step_record = StateRecord(env_time=self.last_env_time,
                                      maze_state=self.last_maze_state,
                                      maze_action=self._last_maze_action_before_skipping,
                                      step_event_log=step_event_log,
                                      reward=reward, done=done, info=info,
                                      serializable_components=self.last_serializable_components)
            self.episode_record.step_records.append(step_record)

            # Collect state and components for the next step
            self._collect_state_and_components()
            self.last_env_time = self.env.get_env_time() if isinstance(self.env,
                                                                       TimeEnvMixin) else self.last_env_time + 1

        return observation, reward, done, info

    @override(BaseEnv)
    def reset(self) -> Any:
        """Record the final state and ship the episode record, reset underlying env and start a new episode record."""
        self._write_episode_record()
        observation = self.env.reset()
        self.last_env_time = self.env.get_env_time() if isinstance(self.env, TimeEnvMixin) else 0
        self._collect_state_and_components()
        self.episode_record = self._build_episode_record()
        return observation

    def render(self, interactive: bool = False, **kwargs) -> None:
        """Render the trajectory data collected during the last step, i.e. MazeState before the last step and
        MazeAction taken during that step.

        :param interactive: If true, builds a (blocking) keyboard-controlled trajectory viewer that allows browsing
          through states in the past steps. Note that this might not work well outside of terminal. Please check
          the note at
          :class:`maze.core.rendering.keyboard_controlled_trajectory_viewer.KeyboardControlledTrajectoryViewer`
          for more information.
        :param kwargs: Optionally, additional arguments that the renderer of the recorded env accepts can
          be passed in."""
        if not isinstance(self.env, RecordableEnvMixin):
            self.env.render()
            return

        assert len(self.episode_record.step_records) > 0, "There are no step records to render (yet?)"
        renderer = self.env.get_renderer()

        if interactive:
            trajectory_viewer = KeyboardControlledTrajectoryViewer(
                episode_record=self.episode_record,
                renderer=renderer,
                renderer_kwargs=kwargs)
            trajectory_viewer.render()
        else:
            last_step_record = self.episode_record.step_records[-1]
            renderer.render(last_step_record.maze_state, last_step_record.maze_action,
                            last_step_record.step_event_log, **kwargs)

    def _collect_state_and_components(self):
        """Collect the state and serializable components, if the env provides access to them.
        """
        if isinstance(self.env, SerializableEnvMixin):
            self.last_serializable_components = self.env.get_serializable_components()
        else:
            self.last_serializable_components = {}

        self.last_maze_state = deepcopy(self.env.get_maze_state())

    def _write_episode_record(self):
        """Records final state of the episode and dispatches the record to trajectory data writers."""
        # Do not record empty episodes
        if not self.episode_record or len(self.episode_record.step_records) == 0:
            return

        # Record the final state
        env_time = self.env.get_env_time() if isinstance(self.env, TimeEnvMixin) else None
        event_collection = EventCollection(self.env.get_step_events() if isinstance(self.env, EventEnvMixin) else [])
        step_event_log = StepEventLog(env_time, events=event_collection)
        final_step_record = StateRecord(
            env_time=env_time,
            maze_state=self.last_maze_state,
            maze_action=None,
            step_event_log=step_event_log,
            reward=None,
            done=None,
            info=None,
            serializable_components=self.last_serializable_components)
        self.episode_record.step_records.append(final_step_record)

        # Write out the current episode
        TrajectoryWriterRegistry.record_trajectory_data(self.episode_record)

    def _build_episode_record(self) -> StateTrajectoryRecord:
        """Build a new episode record with episode ID from the env (if provided) or generated one (if not provided)."""
        if isinstance(self.env, RecordableEnvMixin):
            episode_id = self.env.get_episode_id()
            renderer = self.env.get_renderer()
        else:
            episode_id = str(uuid.uuid4())
            renderer = None

        return StateTrajectoryRecord(episode_id, renderer)

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Keep both actions and observation the same."""
        return self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'TrajectoryRecordingWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        raise RuntimeError("Cloning the TrajectoryRecordingWrapper is not supported.")
