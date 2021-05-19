"""Generate basic statistics for any gym environment."""
import uuid
from typing import Callable, Optional
from typing import TypeVar, Union, Any, Tuple, Dict

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.environment_context import EnvironmentContext
from maze.core.env.event_env_mixin import EventEnvMixin
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.recordable_env_mixin import RecordableEnvMixin
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.time_env_mixin import TimeEnvMixin
from maze.core.events.event_record import EventRecord
from maze.core.log_events.episode_event_log import EpisodeEventLog
from maze.core.log_events.log_events_writer_registry import LogEventsWriterRegistry
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.log_stats.log_stats import LogStatsAggregator, LogStatsLevel, get_stats_logger, LogStatsValue
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.rendering.events_stats_renderer import EventStatsRenderer
from maze.core.wrappers.wrapper import Wrapper


class LogStatsWrapper(Wrapper[MazeEnv], LogStatsEnv):
    """A statistics logging wrapper for :class:`~maze.core.env.base_env.BaseEnv`.

    :param env: The environment to wrap.
    """

    def __init__(self, env: MazeEnv, logging_prefix: Optional[str] = None):
        """Avoid calling this constructor directly, use :method:`wrap` instead."""
        # BaseEnv is a subset of gym.Env
        super().__init__(env)

        # initialize step aggregator
        self.epoch_stats = LogStatsAggregator(LogStatsLevel.EPOCH)
        self.episode_stats = LogStatsAggregator(LogStatsLevel.EPISODE, self.epoch_stats)
        self.step_stats = LogStatsAggregator(LogStatsLevel.STEP, self.episode_stats)

        self.stats_map = {
            LogStatsLevel.EPOCH: self.epoch_stats,
            LogStatsLevel.EPISODE: self.episode_stats,
            LogStatsLevel.STEP: self.step_stats
        }

        if logging_prefix is not None:
            self.epoch_stats.register_consumer(get_stats_logger(logging_prefix))

        self.last_env_time: Optional[int] = None
        self.step_event_log: Optional[StepEventLog] = None
        self.episode_event_log = self._build_episode_event_log()

        self.step_stats_renderer = EventStatsRenderer()

        # get notified at the beginning of a step call (relevant if the step originated
        # within the wrapper hierarchy, e.g. frame skipping)
        if hasattr(env, "context") and isinstance(env.context, EnvironmentContext):
            env.context.register_pre_step(self._pre_step)

    T = TypeVar("T")

    @classmethod
    def wrap(cls, env: T, logging_prefix: Optional[str] = None) -> Union[T, LogStatsEnv]:
        """Creation method providing appropriate type hints. Preferred method to construct the wrapper
        compared to calling the class constructor directly.

        :param env: The environment to be wrapped.
        :param logging_prefix: The episode statistics is connected to the logging system with this tagging
                               prefix. If None, no logging happens.

        :return A newly created wrapper instance.
        """
        return cls(env, logging_prefix)

    @override(BaseEnv)
    def step(self, action: Any) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Collect the rewards for the logging statistics
        """
        assert self.episode_event_log is not None, "Environment must be reset before stepping."

        # get identifier of current substep
        substep_id, _ = self.env.actor_id() if isinstance(self.env, StructuredEnv) else (None, None)

        # take core env step
        obs, rew, done, info = self.env.step(action)

        # record the reward
        self.step_event_log.append(EventRecord(BaseEnvEvents, BaseEnvEvents.reward, dict(value=rew)))

        self._record_stats_if_ready()

        return obs, rew, done, info

    def _pre_step(self) -> None:
        """
        Get notified of a new step method call, just before the events of the previous step are cleared by the
        event system (see :py:meth:`~maze.core.env.environment_context.EnvironmentContext.pre_step`).
        """
        self._record_stats_if_ready()

    def _record_stats_if_ready(self) -> None:
        """Record the event log and statistics from the preceding step, if these are available and ready to record.

        - If the episode event log has not been initialized yet (just after env reset), just performs the initialization
        - Otherwise, checks if stats are ready to record based on env time (for structured envs, we wait till the end
          of the whole structured step) and if so, does the recording
        """
        # If not yet initialized, reset timing data and initialize new step event log for the first step
        if self.episode_event_log is None:
            self.episode_event_log = self._build_episode_event_log()
            self.last_env_time = self.env.get_env_time() if isinstance(self.env, TimeEnvMixin) else 0
            self.step_event_log = StepEventLog(self.last_env_time)
            return

        # Recording of event logs and stats happens:
        #  - for TimeEnvs:   Only if the env time changed, so that we record once per time step
        #  - for other envs: Every step
        if isinstance(self.env, TimeEnvMixin) and self.env.get_env_time() == self.last_env_time:
            return

        self.last_env_time = self.env.get_env_time() if isinstance(self.env, TimeEnvMixin) else self.last_env_time + 1

        if isinstance(self.env, EventEnvMixin):
            self.step_event_log.extend(self.env.get_step_events())

        # add all recorded events to the step aggregator
        for event_record in self.step_event_log.events:
            self.step_stats.add_event(event_record)

        # trigger logging statistics calculation
        self.step_stats.reduce()

        # log raw events and init new step log
        self.episode_event_log.step_event_logs.append(self.step_event_log)
        self.step_event_log = StepEventLog(self.last_env_time)

    @override(BaseEnv)
    def reset(self) -> Any:
        """Reset the environment and trigger the episode statistics calculation of the previous run.
        """
        # generate the episode stats from the previous rollout
        self._calculate_kpis()
        self.episode_stats.reduce()

        self._write_episode_event_log()
        observation = self.env.reset()
        self._record_stats_if_ready()

        return observation

    @override(BaseEnv)
    def close(self):
        """Close the stats rendering figure if needed."""
        self.step_stats_renderer.close()

    @override(LogStatsEnv)
    def get_stats(self, level: LogStatsLevel) -> LogStatsAggregator:
        """Implementation of the LogStatsEnv interface, return the statistics aggregator."""
        aggregator = self.stats_map[level]
        return aggregator

    @override(LogStatsEnv)
    def write_epoch_stats(self):
        """Implementation of the LogStatsEnv interface, call reduce on the episode aggregator.
        """
        if len(self.episode_event_log.step_event_logs) > 0:
            self._calculate_kpis()
            self.episode_stats.reduce()
        self.epoch_stats.reduce()
        self._write_episode_event_log()
        self.episode_event_log = self._build_episode_event_log()

    @override(LogStatsEnv)
    def get_stats_value(self,
                        event: Callable,
                        level: LogStatsLevel,
                        name: Optional[str] = None) -> LogStatsValue:
        """Implementation of the LogStatsEnv interface, obtain the value from the cached aggregator statistics.
        """
        return self.epoch_stats.last_stats[(event, name, None)]

    @override(LogStatsEnv)
    def clear_epoch_stats(self) -> None:
        """Implementation of the LogStatsEnv interface, clear out episode statistics collected so far in this epoch."""
        self.epoch_stats.clear_inputs()

    def render_stats(self,
                     event_name: str = "BaseEnvEvents.reward",
                     metric_name: str = "value",
                     aggregation_func: Optional[Union[str, Callable]] = None,
                     group_by: str = None,
                     post_processing_func: Optional[Union[str, Callable]] = 'cumsum'):
        """Render statistics from the currently running episode.

        Rendering is based on event logs. You can select arbitrary events from those dispatched by the currently
        running environment.

        :param event_name: Name of the even the even log corresponds to
        :param metric_name: Metric to use (one of the event attributes, e.g. "n_items" -- depends on the event type)
        :param aggregation_func: Optionally, specifies how to aggregate the metric on step level, i.e. when there
                                 are multiple same events dispatched during the same step.
        :param group_by: Optionally, another of event attributes to group by on the step level (e.g. "product_id")
        :param post_processing_func: Optionally, a function to post-process the data ("cumsum" is often used)"""
        self.step_stats_renderer.render_current_episode_stats(
            self.episode_event_log, event_name, metric_name,
            aggregation_func, group_by, post_processing_func)

    def _calculate_kpis(self):
        """Calculate KPIs and append them to both aggregated and logged events."""
        if not isinstance(self.env, EventEnvMixin) or len(self.episode_event_log.step_event_logs) == 0:
            return

        kpi_calculator = self.env.get_kpi_calculator()
        if kpi_calculator is None:
            return

        last_maze_state = self.env.get_maze_state() if isinstance(self.env, RecordableEnvMixin) else None

        kpis_dict = kpi_calculator.calculate_kpis(self.episode_event_log, last_maze_state)
        kpi_events = []
        for name, value in kpis_dict.items():
            kpi_events.append(EventRecord(BaseEnvEvents, BaseEnvEvents.kpi, dict(name=name, value=value)))

        for event_record in kpi_events:
            self.episode_stats.add_event(event_record)  # Add the events to episode aggregator
            self.episode_event_log.step_event_logs[-1].events.append(event_record)  # Log the events

    def _write_episode_event_log(self):
        """Send the episode event log to writers."""
        if len(self.episode_event_log.step_event_logs) > 0:
            LogEventsWriterRegistry.record_event_logs(self.episode_event_log)

        self.episode_event_log = None

    def _build_episode_event_log(self) -> EpisodeEventLog:
        """Build a new episode record with episode ID from the env (if provided) or generated one (if not provided)."""
        episode_id = self.env.get_episode_id() if isinstance(self.env, RecordableEnvMixin) else str(uuid.uuid4())
        return EpisodeEventLog(episode_id)

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Keep both actions and observation the same."""
        return self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'LogStatsWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        raise RuntimeError("Cloning the 'LogStatsWrapper' is not supported.")
