""" Contains tests for the step-skip-wrapper. """
from maze.core.log_events.episode_event_log import EpisodeEventLog
from maze.core.log_events.log_events_writer import LogEventsWriter
from maze.core.log_events.monitoring_events import RewardEvents
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.step_skip_wrapper import StepSkipWrapper
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion
from maze.test.shared_test_utils.wrappers import assert_wrapper_clone_from


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
            assert env._steps_done == 1

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

    assert env._steps_done == 0


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


class TestWriter(LogEventsWriter):
    """Testing writer. Keeps the episode event record."""

    def __init__(self):
        self.episode_record = None

    def write(self, episode_record: EpisodeEventLog):
        """Store the record"""
        self.episode_record = episode_record


def test_observation_skipping_wrapper_sticky_flat():
    """ Step skipping unit test """

    n_steps = 3

    # instantiate env
    env = GymMazeEnv("CartPole-v0")
    env = StepSkipWrapper.wrap(env, n_steps=n_steps, skip_mode='sticky')

    # reset environment and run interaction loop
    env.reset()
    cum_rew = 0
    for i in range(2):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        cum_rew += reward

        events = env.get_step_events()
        assert (len([e for e in events if e.interface_method == RewardEvents.reward_original]) == 1)

    assert cum_rew == 6


def test_skipping_wrapper_and_reward_aggregation():
    """ Step skipping unit test """
    observation_conversion = ObservationConversion()

    env = DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion]
    )

    n_steps = 3
    env = StepSkipWrapper.wrap(env, n_steps=n_steps, skip_mode='sticky')

    env.reset()
    for _ in range(4):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        assert(reward == n_steps*10)


def test_skipping_wrapper_clone_from():
    """ Step skipping unit test """

    def make_env():
        env = GymMazeEnv("CartPole-v0")
        env = StepSkipWrapper.wrap(env, n_steps=2, skip_mode="sticky")
        return env

    assert_wrapper_clone_from(make_env, assert_member_list=["_step_actions", "_steps_done"])
