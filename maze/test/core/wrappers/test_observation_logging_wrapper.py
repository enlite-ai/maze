""" Contains tests for the observation logging wrapper. """

import numpy as np

from maze.core.log_events.observation_events import ObservationEvents
from maze.core.wrappers.observation_logging_wrapper import ObservationLoggingWrapper
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict_discrete import \
    DictDiscreteActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion


def build_dummy_structured_environment() -> DummyStructuredEnvironment:
    """
    Instantiates the DummyStructuredEnvironment.

    :return: Instance of a DummyStructuredEnvironment
    """

    observation_conversion = ObservationConversion()

    maze_env = DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictDiscreteActionConversion()],
        observation_conversion=[observation_conversion]
    )

    return DummyStructuredEnvironment(maze_env=maze_env)


def test_observation_logging_wrapper():
    """ Observation logging unit test """

    # instantiate env
    env = build_dummy_structured_environment()

    env = ObservationLoggingWrapper.wrap(env)
    _ = env.reset()
    _ = env.step(env.action_space.sample())[0]

    # test application of wrapper
    for ii in range(2):
        obs = env.step(env.action_space.sample())[0]

        assert len(list(env.core_env.context.event_service.iterate_event_records())) == 3
        for idx, event_record in enumerate(env.core_env.context.event_service.iterate_event_records()):
            assert issubclass(event_record.interface_class, ObservationEvents)
            assert event_record.attributes['step_key'] == ii
            assert event_record.attributes['name'] in ['observation_0', 'observation_1']
            if event_record.interface_method is ObservationEvents.observation_processed:
                if list(obs.keys())[0] in event_record.attributes['name']:
                    obs = obs[list(obs.keys())[0]]
                    print(np.asarray(obs).shape, np.asarray(event_record.attributes['value']).shape)
                    print(np.asarray(obs) - np.asarray(event_record.attributes['value']))
                    assert np.allclose(np.asarray(obs), np.asarray(event_record.attributes['value']))
