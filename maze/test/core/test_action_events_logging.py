""" Contains unit tests for action events logging """
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.log_stats.log_stats import register_log_stats_writer, LogStatsLevel
from maze.utils.log_stats_utils import SimpleStatsLoggingSetup
from maze.core.log_stats.log_stats_writer_console import LogStatsWriterConsole
from maze.core.log_stats.log_stats_writer_tensorboard import LogStatsWriterTensorboard
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


def train(env):
    """ unit test helper function """
    n_episodes = 10
    n_steps_per_episode = 5
    # setup logging
    writer = LogStatsWriterTensorboard(log_dir='test_log', tensorboard_render_figure=True)
    register_log_stats_writer(writer)
    # attach a console writer as well for immediate console feedback
    register_log_stats_writer(LogStatsWriterConsole())

    env = LogStatsWrapper.wrap(env, logging_prefix="train")
    with SimpleStatsLoggingSetup(env):
        for episode in range(n_episodes):
            _ = env.reset()
            for step in range(n_steps_per_episode):
                # take random action
                action = env.action_space.sample()

                # take step in env and trigger log stats writing
                _, _, _, _ = env.step(action)

    # test accessing stats
    env.get_stats(LogStatsLevel.EPOCH)
    env.get_stats_value(BaseEnvEvents.reward, LogStatsLevel.EPOCH, name="mean")


def test_log_action_events_dict_discrete():
    """ action logging unit tests """
    env = build_dummy_maze_env()
    train(env)
