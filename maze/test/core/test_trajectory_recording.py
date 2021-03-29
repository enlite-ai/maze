"""Test recording of trajectory data."""
from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Type

import gym
import numpy as np

from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.events.pubsub import Pubsub
from maze.core.rendering.renderer import Renderer
from maze.core.trajectory_recording.records.raw_maze_state import RawState, RawMazeAction
from maze.core.trajectory_recording.records.trajectory_record import StateTrajectoryRecord
from maze.core.trajectory_recording.writers.trajectory_writer import TrajectoryWriter
from maze.core.trajectory_recording.writers.trajectory_writer_registry import TrajectoryWriterRegistry
from maze.core.wrappers.trajectory_recording_wrapper import TrajectoryRecordingWrapper
from maze.test.shared_test_utils.dummy_env.agents.dummy_policy import DummyGreedyPolicy
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_renderer import DummyRenderer
from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.dummy_env.reward.base import RewardAggregator
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion


def test_records_maze_states_and_actions():
    class CustomDummyRewardAggregator(RewardAggregator):
        """Customized dummy reward aggregator subscribed to BaseEnvEvents."""

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
            self.maze_state = self.observation_space.sample()
            self.pubsub: Pubsub = Pubsub(self.context.event_service)
            self.pubsub.register_subscriber(self.reward_aggregator)
            self.base_event_publisher = self.pubsub.create_event_topic(BaseEnvEvents)
            self.renderer = DummyRenderer()

        def get_renderer(self) -> DummyRenderer:
            """
            Returns DummyRenderer.
            :return: DummyRenderer.
            """
            return self.renderer

        def step(self, maze_action):
            """
            Steps through the environment.
            """
            self.maze_state = self.observation_space.sample()
            self.base_event_publisher.reward(10)
            return super().step(maze_action)

        def get_maze_state(self):
            """
            Returns current state.
            """
            return self.maze_state

        def get_serializable_components(self) -> Dict[str, Any]:
            """
            Returns minimal dict. with components to serialize.
            """
            return {"value": 0}

    class TestWriter(TrajectoryWriter):
        """Mock writer checking the recorded data"""

        def __init__(self):
            self.episode_count = 0
            self.step_count = 0
            self.episode_records = []

        def write(self, episode_record: StateTrajectoryRecord):
            """Count steps and episodes & check instance types"""
            self.episode_records.append(episode_record)
            self.episode_count += 1
            self.step_count += len(episode_record.step_records)

            for step_record in episode_record.step_records[:-1]:
                assert isinstance(step_record.maze_state, dict)
                assert isinstance(step_record.maze_action, dict)
                assert step_record.serializable_components != {}
                assert len(step_record.step_event_log.events) > 0

            final_state_record = episode_record.step_records[-1]
            assert isinstance(final_state_record.maze_state, dict)
            assert final_state_record.maze_action is None
            assert final_state_record.serializable_components != {}

            assert isinstance(episode_record.renderer, Renderer)

    writer = TestWriter()
    TrajectoryWriterRegistry.writers = []  # Ensure there is no other writer
    TrajectoryWriterRegistry.register_writer(writer)

    # env = env_instantiation_example.example_1()
    observation_conversion = ObservationConversion()
    env = DummyEnvironment(
        core_env=CustomDummyCoreEnv(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion]
    )
    env = TrajectoryRecordingWrapper.wrap(env)

    policy = DummyGreedyPolicy()
    states = []  # Observe changes in states over time.

    for _ in range(5):
        obs = env.reset()
        for _ in range(10):
            maze_state = env.get_maze_state()
            states.append(deepcopy(maze_state))
            obs, _, _, _ = env.step(policy.compute_action(observation=obs, maze_state=maze_state, deterministic=True))

    # final env reset required
    env.reset()

    assert writer.step_count == 5 * (10 + 1)  # Count also the recorded final state
    assert writer.episode_count == 5

    # Compare if the recorded inventory changes from the first episode match with the trajectory records
    for step_id in range(10):
        assert np.all(
            (states[step_id][key] == writer.episode_records[0].step_records[step_id].maze_state[key])
            for key in env.observation_conversion.space().spaces
        )


def test_records_trajectory_for_generic_gym_envs():
    class TestWriter(TrajectoryWriter):
        """Mock writer checking the recorded data"""

        def __init__(self):
            self.episode_count = 0
            self.step_count = 0

        def write(self, episode_record: StateTrajectoryRecord):
            """Count steps and episodes & check instance types"""
            self.episode_count += 1
            self.step_count += len(episode_record.step_records)

            for step_record in episode_record.step_records[:-1]:
                assert isinstance(step_record.maze_state, RawState)
                assert isinstance(step_record.maze_action, RawMazeAction)
                assert step_record.serializable_components == {}
                assert len(step_record.step_event_log.events) == 0

            final_state_record = episode_record.step_records[-1]
            assert isinstance(final_state_record.maze_state, RawState)
            assert final_state_record.maze_action is None
            assert final_state_record.serializable_components == {}

            assert episode_record.renderer is None

    writer = TestWriter()
    TrajectoryWriterRegistry.writers = []  # Ensure there is no other writer
    TrajectoryWriterRegistry.register_writer(writer)

    env = gym.make('CartPole-v0')
    env = TrajectoryRecordingWrapper.wrap(env)
    for _ in range(5):
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())
            if i == 9:
                env.render()

    # final env reset required
    env.reset()
    env.close()

    assert writer.step_count == 5 * (10 + 1)  # Count also the recorded final state
    assert writer.episode_count == 5


def test_records_once_per_maze_step_in_multistep_envs():
    """In multi-step envs, trajectory should be recorded once per Maze env step (not in each sub-step)."""

    observation_conversion = ObservationConversion()
    maze_env = DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion]
    )
    env = DummyStructuredEnvironment(maze_env)

    class TestWriter(TrajectoryWriter):
        """Mock writer checking the recorded data"""

        def __init__(self):
            self.episode_count = 0
            self.step_count = 0

        def write(self, episode_record: StateTrajectoryRecord):
            """Count steps and episodes"""
            self.episode_count += 1
            self.step_count += len(episode_record.step_records)

    writer = TestWriter()
    TrajectoryWriterRegistry.writers = []  # Ensure there is no other writer
    TrajectoryWriterRegistry.register_writer(writer)

    env = TrajectoryRecordingWrapper.wrap(env)
    for _ in range(5):
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    # final env reset required
    env.reset()
    env.close()

    # The step count should correspond to core env steps (disregarding sub-steps),
    # I.e., 5 flat steps per 10 structured + 1 final state for each episode
    assert writer.step_count == 5 * (5 + 1)
    assert writer.episode_count == 5
