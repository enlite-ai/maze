"""Test the dict action wrapper"""
import gym
from gym import spaces

from maze.core.wrappers.dict_action_wrapper import DictActionWrapper


class DummyTupleEnv:
    """Dummy testing environment"""

    def __init__(self):
        self.action_space = spaces.Tuple(spaces=[spaces.Discrete(5),
                                                 spaces.MultiBinary(11)])


def test_dict_action_wrapper():
    """ gym env wrapper unit test """
    base_env = gym.make("CartPole-v0")
    env = DictActionWrapper.wrap(base_env)

    assert isinstance(env.action_space, spaces.Dict)

    action = env.action_space.sample()
    out_action = env.action(action)

    assert env.action_space.contains(env.reverse_action(out_action))
    assert env.reverse_action(out_action) == action


def test_tuple_to_dict_action_wrapper():
    """ gym env wrapper unit test """
    base_env = DummyTupleEnv()
    env = DictActionWrapper.wrap(base_env)

    assert isinstance(env.action_space, spaces.Dict)

    action = env.action_space.sample()
    out_action = env.action(action)

    assert env.action_space.contains(env.reverse_action(out_action))
    assert env.reverse_action(out_action) == action
