"""Contains tests for the observation logging wrapper."""

from __future__ import annotations

from maze.core.log_events.monitoring_events import ActionEvents, ObservationEvents, RewardEvents
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.monitoring_wrapper import MazeEnvMonitoringWrapper
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env

import numpy as np


def build_dummy_structured_environment() -> DummyStructuredEnvironment:
    """
    Instantiates the DummyStructuredEnvironment.

    :return: Instance of a DummyStructuredEnvironment
    """

    observation_conversion = ObservationConversion()

    maze_env = DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion],
    )

    return DummyStructuredEnvironment(maze_env=maze_env)


def test_observation_monitoring():
    """Observation logging unit test"""

    # instantiate env
    env = build_dummy_maze_env()

    env = MazeEnvMonitoringWrapper.wrap(env, observation_logging=True, action_logging=False, reward_logging=False)
    env = LogStatsWrapper.wrap(env)  # for accessing events from previous steps
    env.reset()

    # test application of wrapper
    for ii in range(3):
        # Observation will get reported in the next step (when the agent is actually acting on it)
        obs = env.step(env.action_space.sample())[0]

        observation_events = env.get_last_step_events(
            query=[ObservationEvents.observation_original, ObservationEvents.observation_processed]
        )
        assert len(observation_events) == 10
        for event in observation_events:
            assert issubclass(event.interface_class, ObservationEvents)
            obs_name = event.attributes['name']
            assert obs_name in [
                'observation_0',
                'observation_1',
                'action_0_0_mask',
                'action_1_0_mask',
                'action_1_1_mask',
            ]
            if ii > 0:
                assert np.allclose(np.asarray(obs[obs_name]), np.asarray(event.attributes['value']))


def test_reward_monitoring():
    """Reward logging unit test"""

    # instantiate env
    env = build_dummy_maze_env()

    env = MazeEnvMonitoringWrapper.wrap(env, observation_logging=False, action_logging=False, reward_logging=True)
    env = LogStatsWrapper.wrap(env)  # for accessing events from previous steps
    env.reset()
    env.step(env.action_space.sample())

    # test application of wrapper
    for _ in range(2):
        env.step(env.action_space.sample())

        reward_events = env.get_last_step_events(query=[RewardEvents.reward_original, RewardEvents.reward_processed])

        assert len(reward_events) == 2
        for event in reward_events:
            assert issubclass(event.interface_class, RewardEvents)
            assert event.attributes['value'] == 10
            assert event.interface_method in [RewardEvents.reward_original, RewardEvents.reward_processed]


def test_action_monitoring():
    """Action logging unit test"""

    # instantiate env
    env = build_dummy_maze_env()

    env = MazeEnvMonitoringWrapper.wrap(env, observation_logging=False, action_logging=True, reward_logging=False)
    env = LogStatsWrapper.wrap(env)  # for accessing events from previous steps
    env.reset()

    # test application of wrapper
    for _ in range(2):
        env.step(env.action_space.sample())

        action_events = env.get_last_step_events(
            query=[ActionEvents.discrete_action, ActionEvents.continuous_action, ActionEvents.multi_binary_action]
        )

        assert len(action_events) == 7
        for event in action_events:
            if event.attributes['name'] in ['action_0_0', 'action_0_1_0', 'action_0_1_1', 'action_1_0']:
                assert event.interface_method == ActionEvents.discrete_action
            elif event.attributes['name'] in ['action_0_2', 'action_2_0']:
                assert event.interface_method == ActionEvents.continuous_action
            elif event.attributes['name'] in ['action_1_1']:
                assert event.interface_method == ActionEvents.multi_binary_action
            else:
                raise ValueError


def test_monitoring_wrapper_none_obs_skips_obs_events_but_fires_action_and_reward():
    """When obs is None, observation events are skipped while action and reward events are still fired.

    This intentionally produces a count mismatch between obs/action/reward event series for that step.
    This is acceptable because each event type is tracked in an independent queue grouped by
    (step_key, agent_name, name) — there is no positional coupling between them in the stats pipeline.
    A missing obs event simply means that observation metric has no data point for that step.
    """
    env = build_dummy_maze_env()
    monitoring_env = MazeEnvMonitoringWrapper.wrap(
        env, observation_logging=True, action_logging=True, reward_logging=True
    )
    stats_env = LogStatsWrapper.wrap(monitoring_env)
    stats_env.reset()

    # Patch the inner env's step (the env that MonitoringWrapper wraps) to return None as obs
    inner_env = monitoring_env.env
    original_step = inner_env.step

    def patched_step(action):
        _, reward, terminated, truncated, info = original_step(action)
        return None, reward, terminated, truncated, info

    inner_env.step = patched_step
    obs, _, _, _, _ = stats_env.step(stats_env.action_space.sample())

    assert obs is None

    # Observation events must be absent for this step
    observation_events = stats_env.get_last_step_events(
        query=[ObservationEvents.observation_original, ObservationEvents.observation_processed]
    )
    assert len(observation_events) == 0

    # Action and reward events are still fired normally despite obs being None
    action_events = stats_env.get_last_step_events(
        query=[ActionEvents.discrete_action, ActionEvents.continuous_action, ActionEvents.multi_binary_action]
    )
    assert len(action_events) > 0

    reward_events = stats_env.get_last_step_events(query=[RewardEvents.reward_processed])
    assert len(reward_events) > 0

    # Step 2: restore normal obs — observation events must appear again, confirming the stats
    # pipeline is unaffected by the preceding None step (1 out of 2 obs was None).
    inner_env.step = original_step
    obs, _, _, _, _ = stats_env.step(stats_env.action_space.sample())

    assert obs is not None

    observation_events = stats_env.get_last_step_events(
        query=[ObservationEvents.observation_original, ObservationEvents.observation_processed]
    )
    assert len(observation_events) > 0

    action_events = stats_env.get_last_step_events(
        query=[ActionEvents.discrete_action, ActionEvents.continuous_action, ActionEvents.multi_binary_action]
    )
    assert len(action_events) > 0

    reward_events = stats_env.get_last_step_events(query=[RewardEvents.reward_processed])
    assert len(reward_events) > 0
