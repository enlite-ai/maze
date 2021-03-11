""" Contains tests for the observation logging wrapper. """

import numpy as np

from maze.core.log_events.monitoring_events import ObservationEvents, RewardEvents, ActionEvents
from maze.core.wrappers.monitoring_wrapper import MazeEnvMonitoringWrapper
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion


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
    env = build_dummy_structured_environment()

    env = MazeEnvMonitoringWrapper.wrap(env, observation_logging=True, action_logging=False, reward_logging=False)
    _ = env.reset()
    _ = env.step(env.action_space.sample())[0]

    # test application of wrapper
    for ii in range(2):
        obs = env.step(env.action_space.sample())[0]

        assert len(list(env.core_env.context.event_service.iterate_event_records())) == 3
        for idx, event_record in enumerate(env.core_env.context.event_service.iterate_event_records()):
            assert issubclass(event_record.interface_class, ObservationEvents)
            assert event_record.attributes['step_key'] == f"step_key_{ii}"
            assert event_record.attributes['name'] in ['observation_0', 'observation_1']
            if event_record.interface_method is ObservationEvents.observation_processed:
                if list(obs.keys())[0] in event_record.attributes['name']:
                    obs = obs[list(obs.keys())[0]]
                    print(np.asarray(obs).shape, np.asarray(event_record.attributes['value']).shape)
                    print(np.asarray(obs) - np.asarray(event_record.attributes['value']))
                    assert np.allclose(np.asarray(obs), np.asarray(event_record.attributes['value']))


def test_reward_monitoring():
    """ Reward logging unit test """

    # instantiate env
    env = build_dummy_structured_environment()

    env = MazeEnvMonitoringWrapper.wrap(env, observation_logging=False, action_logging=False, reward_logging=True)
    _ = env.reset()
    _ = env.step(env.action_space.sample())[0]

    # test application of wrapper
    for ii in range(2):
        env.step(env.action_space.sample())

        assert len(list(env.core_env.context.event_service.iterate_event_records())) == 1
        for idx, event_record in enumerate(env.core_env.context.event_service.iterate_event_records()):
            assert issubclass(event_record.interface_class, RewardEvents)
            assert event_record.attributes['value'] == 10
            assert event_record.interface_method == RewardEvents.reward_original


def test_action_monitoring():
    """ Action logging unit test """

    # instantiate env
    env = build_dummy_structured_environment()

    env = MazeEnvMonitoringWrapper.wrap(env, observation_logging=False, action_logging=True, reward_logging=False)
    _ = env.reset()
    _ = env.step(env.action_space.sample())[0]

    # test application of wrapper
    for ii in range(2):
        env.step(env.action_space.sample())

        if ii == 0:
            assert len(list(env.core_env.context.event_service.iterate_event_records())) == 4
        if ii == 1:
            assert len(list(env.core_env.context.event_service.iterate_event_records())) == 2

        for idx, event_record in enumerate(env.core_env.context.event_service.iterate_event_records()):
            assert issubclass(event_record.interface_class, ActionEvents)
            assert event_record.attributes['step_key'] == f"step_key_{ii}"

            if event_record.attributes['name'] in ['action_0_0', 'action_0_1_0', 'action_0_1_1', 'action_1_0']:
                assert event_record.interface_method == ActionEvents.discrete_action
            elif event_record.attributes['name'] in ['action_0_2', 'action_2_0']:
                assert event_record.interface_method == ActionEvents.continuous_action
            elif event_record.attributes['name'] in ['action_1_1']:
                assert event_record.interface_method == ActionEvents.multi_binary_action
            else:
                raise ValueError

