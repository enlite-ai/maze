""" Contains tests for the step-skip-wrapper. """
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.step_skip_wrapper import StepSkipWrapper


def test_observation_skipping_wrapper_sticky():
    """ Step skipping unit test """

    # instantiate env
    env = GymMazeEnv("CartPole-v0")
    env = StepSkipWrapper.wrap(env, n_skip_steps=1, skip_mode='sticky')

    # reset environment and run interaction loop
    env.reset()
    cum_rew = 0
    for i in range(2):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        cum_rew += reward

    assert cum_rew == 4
