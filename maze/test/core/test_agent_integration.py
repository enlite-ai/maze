"""Agent integration tests."""

from typing import Any, Tuple, Sequence, Optional

import numpy as np
import pytest

from maze.core.agent.policy import Policy
from maze.core.agent_integration.agent_integration import AgentIntegration
from maze.core.agent_integration.maze_action_candidates import MazeActionCandidates
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.log_events.episode_event_log import EpisodeEventLog
from maze.core.log_events.log_events_writer import LogEventsWriter
from maze.core.log_events.log_events_writer_registry import LogEventsWriterRegistry
from maze.core.log_stats.log_stats import LogStatsWriter, LogStats
from maze.core.log_stats.log_stats import register_log_stats_writer
from maze.core.trajectory_recording.records.trajectory_record import StateTrajectoryRecord
from maze.core.trajectory_recording.writers.trajectory_writer import TrajectoryWriter
from maze.core.trajectory_recording.writers.trajectory_writer_registry import TrajectoryWriterRegistry
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.trajectory_recording_wrapper import TrajectoryRecordingWrapper
from maze.test.shared_test_utils.dummy_env.agents.dummy_policy import DummyGreedyPolicy
from maze.test.shared_test_utils.dummy_env.dummy_renderer import DummyRenderer
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


@pytest.mark.rllib
def test_steps_env_with_single_policy():
    env = build_dummy_maze_env()
    core_env, act_conv, obs_conv = env.core_env, env.action_conversion, env.observation_conversion

    policy = DummyGreedyPolicy()
    agent_integration = AgentIntegration(
        policy=policy,
        action_conversions={0: act_conv},
        observation_conversions={0: obs_conv}
    )

    # Step the environment manually here and query the agent integration wrapper for maze_actions
    maze_state = env.reset()
    reward, done, info = None, None, None

    for i in range(10):
        maze_action = agent_integration.get_maze_action(maze_state, reward, done, info)

        # Compare with the expected maze_action on top of the env that we are stepping
        expected_maze_action = act_conv.space_to_maze(
            policy.compute_action(observation=obs_conv.maze_to_space(maze_state),
                                  maze_state=maze_state, deterministic=True), maze_state
        )
        assert expected_maze_action.keys() == maze_action.keys()
        assert np.all(expected_maze_action[key] == maze_action[key] for key in maze_action.keys())

        maze_state, reward, done, info = core_env.step(maze_action)


@pytest.mark.rllib
def test_handles_multiple_policies():
    """
    Tests whether AgentIntegration handles multiple policies.
    """

    class StaticPolicy(Policy):
        """Mock policy, returns static action provided on initialization."""

        def __init__(self, static_action):
            self.static_action = static_action

        def needs_state(self) -> bool:
            """This policy does not require the state() object to compute the action."""
            return False

        @override(Policy)
        def seed(self, seed: int) -> None:
            """Not applicable since heuristic is deterministic"""
            pass

        def compute_action(self,
                           observation: ObservationType,
                           maze_state: Optional[MazeStateType] = None,
                           env: Optional[BaseEnv] = None,
                           actor_id: ActorID = None,
                           deterministic: bool = False) -> ActionType:
            """Return the set static action"""
            return self.static_action[actor_id.step_key]

        def compute_top_action_candidates(self,
                                          observation: Any,
                                          num_candidates: Optional[int],
                                          maze_state: Optional[MazeStateType] = None,
                                          env: Optional[BaseEnv] = None,
                                          actor_id: ActorID = None,
                                          deterministic: bool = False) -> Tuple[Sequence[Any], Sequence[float]]:
            """Not used"""
            raise NotImplementedError

    env = build_dummy_maze_env()
    core_env, act_conv, obs_conv = env.core_env, env.action_conversion, env.observation_conversion
    static_actions = (act_conv.space().sample(), act_conv.space().sample())

    agent_integration = AgentIntegration(
        policy=StaticPolicy(static_actions),
        action_conversions={0: act_conv, 1: act_conv},
        observation_conversions={0: obs_conv, 1: obs_conv}
    )

    s = env.reset()  # Just get a valid state, the content is not really important
    for i in range(10):
        pol_id = i % 2
        maze_action = agent_integration.get_maze_action(s, 0, False, {}, actor_id=ActorID(pol_id, 0))
        assert maze_action == static_actions[pol_id]  # We should get action from the correct policy


@pytest.mark.rllib
def test_supports_trajectory_recording_wrapper():
    """
    Tests whether agent integration supports trajectory recording wrappers.
    """

    class TestWriter(TrajectoryWriter):
        """Mock writer for checking that trajectory recording goes through."""

        def __init__(self):
            self.step_count = 0

        def write(self, episode_record: StateTrajectoryRecord):
            """Count recorded steps"""
            self.step_count += len(episode_record.step_records)
            assert episode_record.renderer is not None

    step_count = 10

    writer = TestWriter()
    TrajectoryWriterRegistry.writers = []  # Ensure there is no other writer
    TrajectoryWriterRegistry.register_writer(writer)

    env = build_dummy_maze_env()
    core_env, act_conv, obs_conv = env.core_env, env.action_conversion, env.observation_conversion

    policy = DummyGreedyPolicy()
    agent_integration = AgentIntegration(
        policy=policy,
        action_conversions={0: act_conv},
        observation_conversions={0: obs_conv},
        wrapper_types=[TrajectoryRecordingWrapper],
        renderer=DummyRenderer()
    )

    # Step the environment manually here and query the agent integration wrapper for maze_actions
    maze_state = env.reset()
    reward, done, info = None, None, None
    for i in range(step_count):
        maze_action = agent_integration.get_maze_action(maze_state, reward, done, info)
        maze_state, reward, done, info = core_env.step(maze_action)

    # Rollout needs to be finished to notify the wrappers
    agent_integration.finish_rollout(maze_state, reward, done, info)

    assert writer.step_count


@pytest.mark.rllib
def test_logs_events_and_records_stats():
    class TestEventsWriter(LogEventsWriter):
        """Test event writer for checking logged events."""

        def __init__(self):
            self.step_count = 0
            self.reward_events_count = 0

        def write(self, episode_record: EpisodeEventLog):
            """Check that we have some reward events as well as env-specific events."""

            self.step_count += len(episode_record.step_event_logs)
            self.reward_events_count += len(list(episode_record.query_events(BaseEnvEvents.reward)))

    class TestStatsWriter(LogStatsWriter):
        """Test stats writer for checking if stats get calculated."""

        def __init__(self):
            self.collected_stats_count = 0

        def write(self, path: str, step: int, stats: LogStats) -> None:
            """Count number of stats items received"""
            self.collected_stats_count += len(stats)
            pass

    step_count = 10

    # Event logging
    events_writer = TestEventsWriter()
    LogEventsWriterRegistry.writers = []  # Ensure there is no other writer
    LogEventsWriterRegistry.register_writer(events_writer)

    # Stats logging
    stats_writer = TestStatsWriter()
    register_log_stats_writer(stats_writer)

    env = build_dummy_maze_env()
    core_env, act_conv, obs_conv = env.core_env, env.action_conversion, env.observation_conversion

    policy = DummyGreedyPolicy()
    agent_integration = AgentIntegration(
        policy=policy,
        action_conversions={0: act_conv},
        observation_conversions={0: obs_conv},
        wrapper_types=[LogStatsWrapper],
        wrapper_kwargs=[{"logging_prefix": "test"}],
        renderer=DummyRenderer()
    )

    # Step the environment manually here and query the agent integration wrapper for maze_actions
    maze_state = core_env.reset()
    reward, done, info = None, None, None
    for i in range(step_count):
        maze_action = agent_integration.get_maze_action(maze_state, reward, done, info,
                                                        events=list(core_env.get_step_events()))
        state, reward, done, info = core_env.step(maze_action)
        core_env.context.increment_env_step()  # Done by flat env ordinarily

    # Rollout needs to be finished to notify the wrappers
    agent_integration.finish_rollout(maze_state, reward, done, info, events=list(core_env.get_step_events()))

    # Event logging
    assert events_writer.step_count == step_count
    assert events_writer.reward_events_count == step_count

    # Stats logging
    assert stats_writer.collected_stats_count > 0


@pytest.mark.rllib
def test_gets_maze_action_candidates():
    class StaticPolicy(DummyGreedyPolicy):
        """Mock policy, returns static action candidates (careful, always three of them)."""

        def compute_top_action_candidates(self,
                                          observation: ObservationType,
                                          num_candidates: Optional[int],
                                          maze_state: Optional[MazeStateType] = None,
                                          env: Optional[BaseEnv] = None,
                                          actor_id: ActorID = None,
                                          deterministic: bool = False) \
                -> Tuple[Sequence[ActionType], Sequence[float]]:
            """Return static action candidates"""

            return (
                [{"action_0_0": j, "action_1_0": j, "action_1_1": [j % 2] * 5} for j in range(3)],
                [0.95, 0.04, 0.01]
            )

    env = build_dummy_maze_env()
    core_env, act_conv, obs_conv = env.core_env, env.action_conversion, env.observation_conversion

    agent_integration = AgentIntegration(
        policy=StaticPolicy(),
        action_conversions={0: act_conv},
        observation_conversions={0: obs_conv},
        num_candidates=3
    )

    maze_state = core_env.reset()  # Just get a valid state, the content is not really important
    for i in range(10):
        maze_action = agent_integration.get_maze_action(maze_state, 0, False, {})
        assert isinstance(maze_action, MazeActionCandidates)
        assert maze_action.candidates[0]["action_0_0"] == 0
        assert maze_action.candidates[1]["action_0_0"] == 1
        assert maze_action.candidates[2]["action_0_0"] == 2
        assert maze_action.probabilities == [0.95, 0.04, 0.01]


@pytest.mark.rllib
def test_propagates_exceptions_to_main_thread():
    class FailingPolicy(DummyGreedyPolicy):
        """Mock policy, throws an error every time."""

        def compute_action(self,
                           observation: ObservationType,
                           maze_state: Optional[MazeStateType] = None,
                           env: Optional[BaseEnv] = None,
                           actor_id: ActorID = None,
                           deterministic: bool = False) -> ActionType:
            """Throw an error."""
            raise RuntimeError("Test error.")

        def compute_top_action_candidates(self,
                                          observation: ObservationType,
                                          num_candidates: Optional[int],
                                          maze_state: Optional[MazeStateType] = None,
                                          env: Optional[BaseEnv] = None,
                                          actor_id: ActorID = None,
                                          deterministic: bool = False) \
                -> Tuple[Sequence[ActionType], Sequence[float]]:
            """Not used"""

    env = build_dummy_maze_env()
    core_env, act_conv, obs_conv = env.core_env, env.action_conversion, env.observation_conversion

    agent_integration = AgentIntegration(
        policy=FailingPolicy(),
        action_conversions={0: act_conv},
        observation_conversions={0: obs_conv}
    )

    s = core_env.reset()  # Just get a valid state, the content is not really important
    with pytest.raises(RuntimeError) as e_info:
        agent_integration.get_maze_action(s, 0, False, {})
