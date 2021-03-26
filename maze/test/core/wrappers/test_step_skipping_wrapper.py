""" Contains tests for the step-skip-wrapper. """
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.step_skip_wrapper import StepSkipWrapper
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


def assertion_routine(env: StepSkipWrapper) -> None:
    """ Checks if skipping went well. """
    env.reset()

    n_substeps = len(env.action_spaces_dict)
    for idx in range(n_substeps):

        if idx == 1:
            assert env._internal_steps == 1

        if idx == 2:
            assert 0 in env._step_actions
            assert env._step_actions[env.actor_id()[0]] in env.action_space

        # sample action
        action = env.action_space.sample()
        # take env step
        obs, rew, done, info = env.step(action)

        if idx == 0:
            assert rew == 1
        elif idx == 1:
            assert rew == 8

    assert env._internal_steps == 0


def test_observation_skipping_wrapper_sticky():
    """ Step skipping unit test """

    # instantiate env
    env = build_dummy_structured_environment()
    env = StepSkipWrapper.wrap(env, n_steps=6, skip_mode='sticky')

    # test application of wrapper
    assertion_routine(env)


def test_observation_skipping_wrapper_noop():
    """ Step skipping unit test """

    # instantiate env
    env = build_dummy_structured_environment()
    env = StepSkipWrapper.wrap(env, n_steps=6, skip_mode='noop')

    # test application of wrapper
    assertion_routine(env)


def test_observation_skipping_wrapper_sticky_flat():
    """ Step skipping unit test """

    # instantiate env
    env = GymMazeEnv("CartPole-v0")
    env = StepSkipWrapper.wrap(env, n_steps=2, skip_mode='sticky')

    # reset environment and run interaction loop
    env.reset()
    cum_rew = 0
    for i in range(2):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        cum_rew += reward

    assert cum_rew == 4
