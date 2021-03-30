""" Contains tests for the observation logging wrapper. """

import numpy as np

from maze.core.log_events.monitoring_events import ObservationEvents, RewardEvents, ActionEvents
from maze.core.wrappers.monitoring_wrapper import MazeEnvMonitoringWrapper
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


def build_dummy_structured_environment() -> DummyStructuredEnvironment:
    """
    Instantiates the DummyStructuredEnvironment.

    :return: Instance of a DummyStructuredEnvironment
    """

    observation_conversion = ObservationConversion()

    maze_env = DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion]
    )

    return DummyStructuredEnvironment(maze_env=maze_env)


def test_observation_monitoring():
    """ Observation logging unit test """

    # instantiate env
    env = build_dummy_maze_env()

    env = MazeEnvMonitoringWrapper.wrap(env, observation_logging=True, action_logging=False, reward_logging=False)
    env.reset()

    # test application of wrapper
    for ii in range(3):
        # Observation will get reported in the next step (when the agent is actually acting on it)
        obs = env.step(env.action_space.sample())[0]

        assert len(env.core_env.context.event_service.topics[ObservationEvents].events) == 4

        for event in env.core_env.context.event_service.topics[ObservationEvents].events:
            assert issubclass(event.interface_class, ObservationEvents)
            obs_name = event.attributes['name']
            assert obs_name in ['observation_0', 'observation_1']
            if ii > 0:
                assert np.allclose(np.asarray(obs[obs_name]), np.asarray(event.attributes['value']))


def test_reward_monitoring():
    """ Reward logging unit test """

    # instantiate env
    env = build_dummy_maze_env()

    env = MazeEnvMonitoringWrapper.wrap(env, observation_logging=False, action_logging=False, reward_logging=True)
    env.reset()
    env.step(env.action_space.sample())

    # test application of wrapper
    for ii in range(2):
        env.step(env.action_space.sample())

        assert len(env.core_env.context.event_service.topics[RewardEvents].events) == 2
        for event in env.core_env.context.event_service.topics[RewardEvents].events:
            assert issubclass(event.interface_class, RewardEvents)
            assert event.attributes['value'] == 10
            assert event.interface_method in [RewardEvents.reward_original, RewardEvents.reward_processed]


def test_action_monitoring():
    """ Action logging unit test """

    # instantiate env
    env = build_dummy_maze_env()

    env = MazeEnvMonitoringWrapper.wrap(env, observation_logging=False, action_logging=True, reward_logging=False)
    env.reset()

    # test application of wrapper
    for ii in range(2):
        env.step(env.action_space.sample())

        assert len(env.core_env.context.event_service.topics[ActionEvents].events) == 7
        for event in env.core_env.context.event_service.topics[ActionEvents].events:
            if event.attributes['name'] in ['action_0_0', 'action_0_1_0', 'action_0_1_1', 'action_1_0']:
                assert event.interface_method == ActionEvents.discrete_action
            elif event.attributes['name'] in ['action_0_2', 'action_2_0']:
                assert event.interface_method == ActionEvents.continuous_action
            elif event.attributes['name'] in ['action_1_1']:
                assert event.interface_method == ActionEvents.multi_binary_action
            else:
                raise ValueError

