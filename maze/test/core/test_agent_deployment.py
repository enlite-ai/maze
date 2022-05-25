"""Agent integration tests."""

from typing import Tuple, Sequence, Optional

import gym
import numpy as np
import pytest

from maze.core.agent.random_policy import RandomPolicy
from maze.core.agent_deployment.agent_deployment import AgentDeployment
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.log_events.episode_event_log import EpisodeEventLog
from maze.core.log_events.log_events_writer import LogEventsWriter
from maze.core.log_events.log_events_writer_registry import LogEventsWriterRegistry
from maze.core.log_events.monitoring_events import RewardEvents
from maze.core.log_stats.log_stats import LogStatsWriter, LogStats, LogStatsLevel
from maze.core.log_stats.log_stats import register_log_stats_writer
from maze.core.trajectory_recording.records.trajectory_record import StateTrajectoryRecord
from maze.core.trajectory_recording.writers.trajectory_writer import TrajectoryWriter
from maze.core.trajectory_recording.writers.trajectory_writer_registry import TrajectoryWriterRegistry
from maze.core.utils.config_utils import EnvFactory, read_hydra_config
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.trajectory_recording_wrapper import TrajectoryRecordingWrapper
from maze.test.shared_test_utils.dummy_wrappers.step_skip_in_step_wrapper import StepSkipInStepWrapper
from maze.test.shared_test_utils.dummy_env.agents.dummy_policy import DummyGreedyPolicy
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env, \
    build_dummy_structured_env


def test_steps_env_with_single_policy():
    agent_deployment = AgentDeployment(
        policy=DummyGreedyPolicy(),
        env=build_dummy_maze_env()
    )

    # Step the environment manually here and query the agent integration wrapper for maze_actions
    test_policy = DummyGreedyPolicy()
    test_env = build_dummy_maze_env()
    maze_state = test_env.reset()
    reward, done, info = None, None, None

    for i in range(10):
        maze_action = agent_deployment.act(maze_state, reward, done, info)

        # Compare with the expected maze_action on top of the env that we are stepping
        raw_expected_action = test_policy.compute_action(
            observation=test_env.observation_conversion.maze_to_space(maze_state),
            maze_state=maze_state, deterministic=True)
        expected_action = test_env.action_conversion.space_to_maze(raw_expected_action, maze_state
                                                                   )
        assert expected_action.keys() == maze_action.keys()
        assert np.all(expected_action[key] == maze_action[key] for key in maze_action.keys())

        maze_state, reward, done, info = test_env.step(expected_action)


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

    agent_deployment = AgentDeployment(
        policy=DummyGreedyPolicy(),
        env=TrajectoryRecordingWrapper.wrap(build_dummy_maze_env()),
    )

    # Step the environment manually here and query the agent integration wrapper for maze_actions
    test_core_env = build_dummy_maze_env().core_env
    maze_state = test_core_env.reset()
    reward, done, info = None, None, None
    for i in range(10):
        maze_action = agent_deployment.act(maze_state, reward, done, info)
        maze_state, reward, done, info = test_core_env.step(maze_action)

    # Rollout needs to be finished to notify the wrappers
    agent_deployment.close(maze_state, reward, done, info)

    assert writer.step_count == step_count + 1  # count terminal state as well


def test_supports_multi_step_wrappers():
    env = build_dummy_structured_env()
    env = LogStatsWrapper.wrap(env)
    agent_deployment = AgentDeployment(
        policy=DummyGreedyPolicy(),
        env=env
    )

    # Step the environment manually here and query the agent integration wrapper for maze_actions
    test_core_env = build_dummy_structured_env().core_env
    maze_state = test_core_env.reset()
    reward, done, info = 0, False, {}

    for i in range(4):
        maze_action = agent_deployment.act(maze_state, reward, done, info)
        maze_state, reward, done, info = test_core_env.step(maze_action)

    agent_deployment.close(maze_state, reward, done, info)
    assert env.get_stats_value(
        BaseEnvEvents.reward,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 4  # Step count is still 4 event with multiple sub-steps, as it is detected based on env time

    assert env.get_stats_value(
        RewardEvents.reward_original,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 4


def test_supports_step_skipping_wrappers():
    env = build_dummy_maze_env()
    env = StepSkipInStepWrapper.wrap(env)
    env = LogStatsWrapper.wrap(env)
    agent_deployment = AgentDeployment(
        policy=DummyGreedyPolicy(),
        env=env
    )

    # Step the environment manually here and query the agent integration wrapper for maze_actions
    test_core_env = build_dummy_maze_env().core_env
    maze_state = test_core_env.reset()
    reward, done, info = 0, False, {}

    for i in range(4):
        maze_action = agent_deployment.act(maze_state, reward, done, info)
        maze_state, reward, done, info = test_core_env.step(maze_action)

    agent_deployment.close(maze_state, reward, done, info)
    assert env.get_stats_value(
        BaseEnvEvents.reward,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 2

    assert env.get_stats_value(
        RewardEvents.reward_original,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 4


def test_records_stats():
    env = LogStatsWrapper.wrap(build_dummy_maze_env())
    agent_deployment = AgentDeployment(
        policy=DummyGreedyPolicy(),
        env=env
    )

    # Step the environment manually here and query the agent integration wrapper for maze_actions
    test_core_env = build_dummy_maze_env().core_env
    maze_state = test_core_env.reset()
    reward, done, info = 0, False, {}

    for i in range(5):
        maze_action = agent_deployment.act(maze_state, reward, done, info)
        maze_state, reward, done, info = test_core_env.step(maze_action)

    agent_deployment.close(maze_state, reward, done, info)
    assert env.get_stats_value(
        RewardEvents.reward_original,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 5

    assert env.get_stats_value(
        BaseEnvEvents.reward,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 5


def test_writes_event_and_stats_logs():
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

    agent_deployment = AgentDeployment(
        policy=DummyGreedyPolicy(),
        env=build_dummy_maze_env(),
        wrappers={LogStatsWrapper: {"logging_prefix": "test"}}
    )

    # Step the environment manually here and query the agent integration wrapper for maze_actions
    test_core_env = build_dummy_maze_env().core_env
    maze_state = test_core_env.reset()
    reward, done, info = None, None, None
    for i in range(step_count):
        maze_action = agent_deployment.act(maze_state, reward, done, info,
                                           events=list(test_core_env.get_step_events()))
        state, reward, done, info = test_core_env.step(maze_action)
        test_core_env.context.increment_env_step()  # Done by maze env ordinarily

    # Rollout needs to be finished to notify the wrappers
    agent_deployment.close(maze_state, reward, done, info, events=list(test_core_env.get_step_events()))

    # Event logging
    assert events_writer.step_count == step_count
    assert events_writer.reward_events_count == step_count

    # Stats logging
    assert stats_writer.collected_stats_count > 0


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

        def compute_top_action_candidates(self, observation: ObservationType, num_candidates: Optional[int],
                                          maze_state: Optional[MazeStateType], env: Optional[BaseEnv],
                                          actor_id: ActorID = None) \
                -> Tuple[Sequence[ActionType], Sequence[float]]:
            """Not used"""

    agent_deployment = AgentDeployment(
        policy=FailingPolicy(),
        env=build_dummy_maze_env()
    )

    test_core_env = build_dummy_maze_env().core_env
    s = test_core_env.reset()  # Just get a valid state, the content is not really important
    with pytest.raises(RuntimeError) as e_info:
        agent_deployment.act(s, 0, False, {})


def test_configures_from_hydra():
    # Note: Needs to be `conf_rollout`, so the policy config is present as well
    cfg = read_hydra_config(config_module="maze.conf", config_name="conf_rollout", env="gym_env")

    agent_deployment = AgentDeployment(
        policy=cfg.policy,
        env=cfg.env,
        wrappers=cfg.wrappers
    )

    external_env = EnvFactory(cfg.env, wrappers={})().core_env

    maze_state = external_env.reset()
    reward, done, info = 0, False, {}

    for i in range(10):
        maze_action = agent_deployment.act(maze_state, reward, done, info)
        maze_state, reward, done, info = external_env.step(maze_action)

    agent_deployment.close(maze_state, reward, done, info)
    assert agent_deployment.env.get_stats_value(
        BaseEnvEvents.reward,
        LogStatsLevel.EPOCH,
        name="total_step_count"
    ) == 10


def test_works_with_gym_maze_envs():
    env = GymMazeEnv("CartPole-v0")
    policy = RandomPolicy(action_spaces_dict=env.action_spaces_dict)

    agent_deployment = AgentDeployment(
        policy=policy,
        env=env
    )

    external_env = gym.make("CartPole-v0")

    maze_state = external_env.reset()
    reward, done, info = 0, False, {}

    for i in range(10):
        maze_action = agent_deployment.act(maze_state, reward, done, info)
        maze_state, reward, done, info = external_env.step(maze_action)

    agent_deployment.close(maze_state, reward, done, info)
