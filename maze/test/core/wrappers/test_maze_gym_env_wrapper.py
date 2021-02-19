""" Contains unit test for the maze gym environment wrapper """
import gym
import numpy as np

from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv


def test_maze_gym_env_wrapper():
    """ gym env wrapper unit test """
    env = GymMazeEnv(env="CartPole-v0")
    env.seed(1234)
    obs = env.reset()
    env.observation_conversion.space_to_maze(obs)
    assert not env.is_actor_done()
    assert env.get_serializable_components() == {}
    for _ in range(10):
        env.step(env.action_space.sample())
    env.close()


def test_multi_step_dict_gym_env():
    env = GymMazeEnv(env="CartPole-v0")
    assert isinstance(env.action_spaces_dict[0], gym.spaces.Dict)
    assert isinstance(env.observation_spaces_dict[0], gym.spaces.Dict)


def test_gets_formatted_actions_and_observations():
    gym_env = gym.make("CartPole-v0")
    gym_obs = gym_env.reset()
    gym_act = gym_env.action_space.sample()

    wrapped_env = GymMazeEnv(env="CartPole-v0")
    wrapped_env.seed(1234)
    assert not wrapped_env.is_actor_done()
    assert wrapped_env.actor_id() == (0, 0)
    obs_dict, act_dict = wrapped_env.get_observation_and_action_dicts(gym_obs, gym_act, False)
    assert np.all(gym_obs.astype(np.float32) == obs_dict[0]["observation"])
    assert np.all(gym_act == act_dict[0]["action"])
    wrapped_env.close()
