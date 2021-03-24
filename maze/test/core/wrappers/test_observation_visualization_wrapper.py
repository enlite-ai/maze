""" Test observation visualization wrapper """
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.observation_visualization_wrapper import ObservationVisualizationWrapper
from maze.utils.log_stats_utils import SimpleStatsLoggingSetup


def test_observation_monitoring():
    """ Observation logging unit test """
    env = GymMazeEnv(env="CartPole-v0")

    env = ObservationVisualizationWrapper.wrap(env, plot_function=None)
    env = LogStatsWrapper.wrap(env, logging_prefix="train")

    with SimpleStatsLoggingSetup(env, log_dir="."):
        env.reset()
        done = False
        while not done:
            obs, rew, done, info = env.step(env.action_space.sample())
