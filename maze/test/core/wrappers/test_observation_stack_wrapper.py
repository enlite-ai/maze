""" Contains tests for the observation stacking wrappers. """
from copy import deepcopy
from typing import Type, Any

import maze.test.core.wrappers as wrapper_module
from maze.core.wrappers.observation_stack_by_actor_id_wrapper import ObservationStackByActorIDWrapper
from maze.core.wrappers.observation_stack_wrapper import ObservationStackWrapper
from maze.test.shared_test_utils.config_testing_utils import load_env_config
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict_discrete import \
    DictDiscreteActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion
import pytest


def build_dummy_maze_environment() -> DummyEnvironment:
    """Instantiates a dummy Maze env."""
    observation_conversion = ObservationConversion()

    return DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictDiscreteActionConversion()],
        observation_conversion=[observation_conversion]
    )


def build_dummy_structured_environment() -> DummyStructuredEnvironment:
    """Instantiates the DummyStructuredEnvironment."""
    return DummyStructuredEnvironment(maze_env=build_dummy_maze_environment())


def get_n_stacked_obs(
        wrapper_class: Type[Any],
        env: ObservationStackWrapper | ObservationStackByActorIDWrapper,
        obs_key: str
) -> int:
    """ Return the number of stacked observations
    :param wrapper_class: The observation wrapper class used
    :param env: The environment
    :param obs_key: the observation key to check
    :return: The number of stacked observations labeled obs_key
    """
    if wrapper_class == ObservationStackWrapper:
        return len(env._observation_stack[obs_key])
    if wrapper_class == ObservationStackByActorIDWrapper:
        return len(env._observation_stack[env.actor_id()][obs_key])

    raise NotImplementedError('wrapper_class should be in [ObservationStackWrapper, ObservationStackByActorIDWrapper]')


def assertion_routine(env: ObservationStackWrapper | ObservationStackByActorIDWrapper) -> None:
    """ Checks if stacking went well. """
    # test application of wrapper
    obs = env.reset()

    for _ in range(3):

        observation_keys = list(obs.keys())
        for key in ['observation_0']:
            assert key in observation_keys
            assert obs[key] in env.observation_spaces_dict[0][key]
        obs = env.step(env.action_space.sample())[0]

        observation_keys = list(obs.keys())
        for key in ['observation_1', 'observation_1-stacked']:
            assert key in observation_keys
            assert obs[key] in env.observation_spaces_dict[1][key]
        obs = env.step(env.action_space.sample())[0]


@pytest.mark.parametrize('wrapper_class', [ObservationStackWrapper, ObservationStackByActorIDWrapper])
def test_observation_stack_wrapper(wrapper_class):
    """ Observation stacking unit test """

    # instantiate env
    env = build_dummy_structured_environment()

    # wrapper config
    config = {
        "stack_config": [
            {"observation": "observation_0",
             "keep_original": False,
             "tag": None,
             "delta": True,
             "stack_steps": 2},
            {"observation": "observation_1",
             "keep_original": True,
             "tag": "stacked",
             "delta": False,
             "stack_steps": 2}
        ]
    }

    env = wrapper_class.wrap(env, stack_config=config["stack_config"])

    # test application of wrapper
    assertion_routine(env)


@pytest.mark.parametrize('wrapper_class', [ObservationStackWrapper, ObservationStackByActorIDWrapper])
def test_observation_stack_init_from_yaml_config(wrapper_class):
    """ Pre-processor unit test """

    # load config
    config = load_env_config(wrapper_module, "dummy_observation_stack_config_file.yml")

    # init environment
    env = build_dummy_structured_environment()
    env = wrapper_class(env, **config["observation_stack_wrapper"])

    # test application of wrapper
    assertion_routine(env)


@pytest.mark.parametrize('wrapper_class', [ObservationStackWrapper, ObservationStackByActorIDWrapper])
def test_observation_stack_wrapper_nothing_to_stack(wrapper_class):
    """ Observation stacking unit test """

    # instantiate env
    env = build_dummy_structured_environment()

    # wrapper config
    config = {
        "stack_config": [
            {"observation": "observation_0",
             "keep_original": False,
             "tag": None,
             "delta": True,
             "stack_steps": 1},
            {"observation": "observation_1",
             "keep_original": True,
             "tag": "stacked",
             "delta": False,
             "stack_steps": 1}
        ]
    }

    env = wrapper_class.wrap(env, stack_config=config["stack_config"])
    obs = env.reset()
    assert obs["observation_0"].shape == (3, 32, 32)


@pytest.mark.parametrize('wrapper_class', [ObservationStackWrapper, ObservationStackByActorIDWrapper])
def test_stack_reset_on_trajectory_load(wrapper_class: Type[Any]):
    env = build_dummy_maze_environment()

    # wrapper config
    stack_config = [{
        "observation": "observation_0",
        "keep_original": False,
        "tag": "stacked",
        "delta": False,
        "stack_steps": 2
    }]
    env = wrapper_class.wrap(env, stack_config=stack_config)

    # == Check that stepping affects the stack as expected ==

    # Stack should be empty at the beginning
    assert get_n_stacked_obs(wrapper_class, env, "observation_0") == 0

    # Env reset puts the first observation on stack
    env.reset()
    assert get_n_stacked_obs(wrapper_class, env, "observation_0") == 1

    # Env step puts second observation on stack
    env.step(env.action_space.sample())
    assert get_n_stacked_obs(wrapper_class, env, "observation_0") == 2

    # == Check loading trajectory data ==

    maze_state = env.get_maze_state()
    maze_action = env.action_conversion.space_to_maze(env.action_space.sample(), maze_state)

    # Loading the first step in the episode should reduce the stack back to 1 (like after a reset)
    env.get_observation_and_action_dicts(deepcopy(maze_state), deepcopy(maze_action), first_step_in_episode=True)
    assert get_n_stacked_obs(wrapper_class, env, "observation_0") == 1

    # Loading the next step should put second observation on stack
    env.get_observation_and_action_dicts(deepcopy(maze_state), deepcopy(maze_action), first_step_in_episode=False)
    assert get_n_stacked_obs(wrapper_class, env, "observation_0") == 2

    # Loading first step of another episode should reduce the stack back to 1
    env.get_observation_and_action_dicts(deepcopy(maze_state), deepcopy(maze_action), first_step_in_episode=True)
    assert get_n_stacked_obs(wrapper_class, env, "observation_0") == 1
