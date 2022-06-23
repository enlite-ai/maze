"""Test rollouts run using the rollout runners"""

from typing import Dict

import pytest

from maze.core.agent.dummy_cartpole_policy import DummyCartPolePolicy
from maze.core.agent.heuristic_lunar_lander_policy import HeuristicLunarLanderPolicy
from maze.core.agent.replay_recorded_actions_policy import ReplayRecordedActionsPolicy
from maze.core.rollout.action_record_rollout_runner import ActionRecordRolloutRunner
from maze.core.rollout.parallel_rollout_runner import ParallelRolloutRunner
from maze.core.rollout.sequential_rollout_runner import SequentialRolloutRunner
from maze.core.utils.seeding import MazeSeeding
from maze.core.wrappers.action_recording_wrapper import ActionRecordingWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.monitoring_wrapper import MazeEnvMonitoringWrapper
from maze.test.shared_test_utils.run_maze_utils import run_maze_job

heuristic_rollouts = [
    {"runner": "sequential"},
    {"runner": "parallel"}
]

# Ensure we are running test configuration and no wrappers (as we do not have the stats
# to initialize observation normalization with)
heuristic_rollouts_defaults = dict(env="gym_env", configuration="test")
heuristic_rollouts = [pytest.param({**heuristic_rollouts_defaults, **r}, id=r['runner']) for r in heuristic_rollouts]


@pytest.mark.parametrize("hydra_overrides", heuristic_rollouts)
def test_heuristic_rollouts(hydra_overrides: Dict[str, str]):
    """Runs rollout of a dummy policy on cartpole using the sequential and parallel runners."""
    run_maze_job(hydra_overrides, config_module="maze.conf", config_name="conf_rollout")


def test_rollouts_from_python():
    env, agent = GymMazeEnv("CartPole-v0"), DummyCartPolePolicy()

    sequential = SequentialRolloutRunner(
        n_episodes=2,
        max_episode_steps=2,
        deterministic=False,
        record_trajectory=False,
        record_event_logs=False,
        render=False)
    sequential.maze_seeding = MazeSeeding(env_seed=1234, agent_seed=4321, cudnn_determinism_flag=False,
                                          explicit_env_seeds=None, explicit_agent_seeds=None, shuffle_seeds=False)
    sequential.run_with(env=env, wrappers={}, agent=agent)

    parallel = ParallelRolloutRunner(
        n_episodes=2,
        max_episode_steps=2,
        deterministic=False,
        record_trajectory=False,
        record_event_logs=False,
        n_processes=2)
    parallel.maze_seeding = MazeSeeding(env_seed=1234, agent_seed=4321, cudnn_determinism_flag=False,
                                        explicit_env_seeds=None, explicit_agent_seeds=None, shuffle_seeds=False)
    # Test with a wrapper config as well
    parallel.run_with(
        env=env,
        wrappers={MazeEnvMonitoringWrapper: {"observation_logging": True,
                                             "action_logging": False,
                                             "reward_logging": False}},
        agent=agent)


def test_sequential_rollout_with_rendering():
    env, agent = GymMazeEnv("CartPole-v0"), DummyCartPolePolicy()
    sequential = SequentialRolloutRunner(
        n_episodes=2,
        max_episode_steps=2,
        deterministic=False,
        record_trajectory=True,
        record_event_logs=False,
        render=True)
    sequential.maze_seeding = MazeSeeding(env_seed=1234, agent_seed=4321, cudnn_determinism_flag=False,
                                          explicit_env_seeds=None, explicit_agent_seeds=None, shuffle_seeds=False)
    sequential.run_with(env=env, wrappers={}, agent=agent)


def test_action_record_rollout():
    teacher_policy = HeuristicLunarLanderPolicy()

    env = GymMazeEnv("LunarLander-v2")
    env = ActionRecordingWrapper.wrap(env, record_maze_actions=False, record_actions=True,
                                      output_dir="action_records")

    env.seed(1234)
    obs = env.reset()
    done = False
    while not done:
        action = teacher_policy.compute_action(obs)
        obs, rew, done, info = env.step(action)

        if done:
            env.seed(1234)
            env.reset()

    env = GymMazeEnv("LunarLander-v2")

    runner = ActionRecordRolloutRunner(max_episode_steps=10000,
                                       deterministic=False,
                                       action_record_path="action_records",
                                       normalization_samples=10,
                                       n_processes=2,
                                       verbose=False)

    runner.run_with(
        env=env,
        wrappers={MazeEnvMonitoringWrapper: {"observation_logging": True,
                                             "action_logging": False,
                                             "reward_logging": False}},
        agent=ReplayRecordedActionsPolicy(action_record_path=None, with_agent_actions=True))
