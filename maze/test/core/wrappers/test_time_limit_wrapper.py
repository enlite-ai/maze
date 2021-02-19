""" Contains tests for the time limit wrapper. """
from maze.core.wrappers.time_limit_wrapper import TimeLimitWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_base_env, build_dummy_maze_env


def test_time_limit_wrapper():
    """ time limit wrapper unit tests """
    env = build_dummy_maze_env()
    env = TimeLimitWrapper.wrap(env, max_episode_steps=5)
    env.set_max_episode_steps(max_episode_steps=5)

    env.seed(1234)
    env.reset()
    for i in range(5):
        obs, rew, done, info = env.step(env.action_space.sample())
        if i >= 4:
            assert done
    env.close()


def test_time_limit_wrapper_with_spec():
    """ time limit wrapper unit tests """

    class Spec:
        def __init__(self):
            self.max_episode_steps = 5

    spec = Spec()

    env = build_dummy_maze_env()
    env.__setattr__("spec", spec)
    env = TimeLimitWrapper.wrap(env, max_episode_steps=None)

    env.seed(1234)
    env.reset()
    for i in range(5):
        obs, rew, done, info = env.step(env.action_space.sample())
        if i >= 4:
            assert done
    env.close()


def test_time_limit_wrapper_time_env():
    """ time limit wrapper unit tests """
    env = build_dummy_base_env()
    env = TimeLimitWrapper.wrap(env, max_episode_steps=5)

    env.seed(1234)
    env.reset()
    for i in range(5):
        obs, rew, done, info = env.step(None)
        if i >= 4:
            assert done
    env.close()
