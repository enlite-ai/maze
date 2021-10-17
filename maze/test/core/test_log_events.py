"""Test event logging."""
import random
from abc import ABC
from typing import Tuple, Optional
from typing import Union, List, Type, Dict

import gym
import numpy as np

from maze.core.env.base_env import BaseEnv
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.maze_state import MazeStateType
from maze.core.env.time_env_mixin import TimeEnvMixin
from maze.core.events.pubsub import Pubsub
from maze.core.log_events.episode_event_log import EpisodeEventLog
from maze.core.log_events.kpi_calculator import KpiCalculator
from maze.core.log_events.log_events_writer import LogEventsWriter
from maze.core.log_events.log_events_writer_registry import LogEventsWriterRegistry
from maze.core.log_stats.event_decorators import define_episode_stats, define_step_stats, define_epoch_stats
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.dummy_env.reward.base import RewardAggregator
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion as \
    DummyActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion \
    as DummyObservationConversion


def _run_rollout_maze(env: Union[BaseEnv, gym.Env], n_steps_per_episode: int, n_episodes: int, writer: LogEventsWriter):
    LogEventsWriterRegistry.writers = []  # Ensure there is no other writer
    LogEventsWriterRegistry.register_writer(writer)

    env = LogStatsWrapper.wrap(env)
    for _ in range(n_episodes):
        env.reset()
        for i in range(n_steps_per_episode):
            env.step(env.action_space.sample())
            if i == n_steps_per_episode - 1:
                env.render_stats()
    env.close()
    env.write_epoch_stats()  # Test ending without env reset


def test_logs_events():
    class CustomDummyKPICalculator(KpiCalculator):
        """
        Dummy KPIs for dummy environment.
        """

        def calculate_kpis(self, episode_event_log: EpisodeEventLog, last_maze_state: MazeStateType) -> Dict[
            str, float]:
            """
            Returns a dummy KPI.
            """
            return {"dummy_kpi": random.random()}

    class CustomDummyRewardAggregator(RewardAggregator):
        """
        Customized dummy reward aggregator subscribed to BaseEnvEvents.
        """

        def get_interfaces(self) -> List[Type[ABC]]:
            """
            Return events class is subscribed to.
            """
            additional_interfaces: List[Type[ABC]] = [BaseEnvEvents]
            parent_interfaces = super().get_interfaces()
            return additional_interfaces + parent_interfaces

    class CustomDummyCoreEnv(DummyCoreEnvironment):
        """
        Customized dummy core env with serializable components that regenerates state only in step.
        """

        def __init__(self, observation_space):
            super().__init__(observation_space)
            self.reward_aggregator = CustomDummyRewardAggregator()
            self.pubsub: Pubsub = Pubsub(self.context.event_service)
            self.pubsub.register_subscriber(self.reward_aggregator)
            self.base_event_publisher = self.pubsub.create_event_topic(BaseEnvEvents)
            self.kpi_calculator = CustomDummyKPICalculator()

        def get_kpi_calculator(self) -> CustomDummyKPICalculator:
            """KPIs are supported."""
            return self.kpi_calculator

    class TestWriter(LogEventsWriter):
        """Mock writer checking the logged events"""

        def __init__(self):
            self.episode_count = 0
            self.step_count = 0

        def write(self, episode_record: EpisodeEventLog):
            """Count steps and episodes & check that some events were logged in each step"""
            self.episode_count += 1
            self.step_count += len(episode_record.step_event_logs)

            for step_id, step_event_log in enumerate(episode_record.step_event_logs):
                assert step_id == step_event_log.env_time
                assert len(step_event_log.events) > 0

            last_step_event_log = episode_record.step_event_logs[-1]
            kpi_count_in_last_step = len(list(last_step_event_log.events.query_events(BaseEnvEvents.kpi)))
            assert kpi_count_in_last_step != 0
            # KPIs should be a part of the last step record only
            assert kpi_count_in_last_step == len(list(episode_record.query_events(BaseEnvEvents.kpi)))

    observation_conversion = ObservationConversion()
    writer = TestWriter()
    _run_rollout_maze(
        env=DummyEnvironment(
            core_env=CustomDummyCoreEnv(observation_conversion.space()),
            action_conversion=[DictActionConversion()],
            observation_conversion=[observation_conversion]
        ),
        n_episodes=5,
        n_steps_per_episode=10,
        writer=writer)

    assert writer.step_count == 5 * 10
    assert writer.episode_count == 5


def test_logs_events_for_generic_gym_envs():
    class TestWriter(LogEventsWriter):
        """Mock writer checking the logged events"""

        def __init__(self):
            self.episode_count = 0
            self.step_count = 0

        def write(self, episode_record: EpisodeEventLog):
            """Count steps and episodes & check that some events were logged in each step"""
            self.episode_count += 1
            self.step_count += len(episode_record.step_event_logs)

            for step_id, step_event_log in enumerate(episode_record.step_event_logs):
                assert step_id == step_event_log.env_time

    writer = TestWriter()
    _run_rollout_maze(
        env=GymMazeEnv(gym.make('CartPole-v0')),
        n_episodes=5,
        n_steps_per_episode=10,
        writer=writer)

    assert writer.step_count == 5 * 10
    assert writer.episode_count == 5


def test_logs_custom_env_time():
    class CustomTimedDummyEnv(DummyEnvironment, TimeEnvMixin):
        """A subclass of the dummy env that has custom env time."""

        def reset(self):
            """Start counting env time from 1337."""
            obs = super().reset()
            self.core_env.context.step_id = 1337
            return obs

    dummy_observation_conversion = DummyObservationConversion()
    core_env = DummyCoreEnvironment(observation_space=dummy_observation_conversion.space())

    env = CustomTimedDummyEnv(
        core_env=core_env,
        action_conversion=[DummyActionConversion()],
        observation_conversion=[dummy_observation_conversion]
    )

    class TestWriter(LogEventsWriter):
        """Mock writer checking the logged events"""

        def __init__(self):
            self.episode_count = 0
            self.step_count = 0

        def write(self, episode_record: EpisodeEventLog):
            """Count steps and episodes & check that env time is available and set to the correct value"""
            self.episode_count += 1
            self.step_count += len(episode_record.step_event_logs)

            for step_id, step_event_log in enumerate(episode_record.step_event_logs):
                assert step_event_log.env_time == 1337 + step_id

    writer = TestWriter()
    _run_rollout_maze(
        env=env,
        n_episodes=5,
        n_steps_per_episode=10,
        writer=writer)

    assert writer.step_count == 5 * 10
    assert writer.episode_count == 5


def test_records_once_per_maze_step_in_multistep_envs():
    """In multi-step envs, events should be recorded once per Maze env step (not in each sub-step)."""

    class _SubStepEvents(ABC):
        @define_epoch_stats(sum)
        @define_episode_stats(len)
        @define_step_stats(len)
        def sub_step_event(self):
            """Dispatched in each sub step."""

    class _CoreEnvEvents(ABC):
        @define_epoch_stats(sum)
        @define_episode_stats(len)
        @define_step_stats(len)
        def core_env_step_event(self):
            """Dispatched when core env steps (only once per step)."""

    class EventDummyEnv(DummyCoreEnvironment):
        """Dummy core env, dispatches one step event in every step."""

        def __init__(self, observation_space: gym.spaces.space.Space):
            super().__init__(observation_space)
            self.dummy_events = self.pubsub.create_event_topic(_CoreEnvEvents)

        def step(self, maze_action: Dict) -> Tuple[Dict[str, np.ndarray], float, bool, Optional[Dict]]:
            """Dispatch the step event..."""
            self.dummy_events.core_env_step_event()
            return super().step(maze_action)

    class DummyMultiStepEnv(DummyStructuredEnvironment):
        """Simulates simple two-step env. The underlying env is stepped only in the second step."""

        def __init__(self, maze_env):
            super().__init__(maze_env)
            self.dummy_events = self.pubsub.create_event_topic(_SubStepEvents)

        def _action0(self, action) -> Tuple[Dict, float, bool, Optional[Dict[str, np.ndarray]]]:
            self.dummy_events.sub_step_event()
            return {}, 0, False, None

        def _action1(self, action) -> Tuple[Dict, float, bool, Optional[Dict[str, np.ndarray]]]:
            self.dummy_events.sub_step_event()
            return self.maze_env.step(action)

    class TestWriter(LogEventsWriter):
        """Testing writer. Keeps the episode event record."""

        def __init__(self):
            self.episode_record = None

        def write(self, episode_record: EpisodeEventLog):
            """Store the record"""
            self.episode_record = episode_record

    # Init the env hierarchy
    observation_conversion = ObservationConversion()
    maze_env = DummyEnvironment(
        core_env=EventDummyEnv(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion]
    )
    env = DummyMultiStepEnv(maze_env)

    # Run the rollout
    writer = TestWriter()
    _run_rollout_maze(
        env=env,
        n_episodes=1,
        n_steps_per_episode=10,
        writer=writer)

    # There should be one core env step event and two substep events recorded in every step.
    assert writer.episode_record is not None
    for step_id, step_event_log in enumerate(writer.episode_record.step_event_logs):
        if (step_id + 1) % 2 == 0:
            assert len(step_event_log.events.query_events(_CoreEnvEvents.core_env_step_event)) == 1
        assert len(step_event_log.events.query_events(_SubStepEvents.sub_step_event)) == 2
        assert step_event_log.env_time == step_id
