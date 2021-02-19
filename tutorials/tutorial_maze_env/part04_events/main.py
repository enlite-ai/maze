""" Test script CoreEnv """
from maze.utils.log_stats_utils import SimpleStatsLoggingSetup
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from tutorials.tutorial_maze_env.part04_events.env.maze_env import maze_env_factory


def main():
    # init maze environment including observation and action interfaces
    env = maze_env_factory(max_pieces_in_inventory=200,
                           raw_piece_size=[100, 100],
                           static_demand=(30, 15))

    # wrap environment with logging wrapper
    env = LogStatsWrapper(env, logging_prefix="main")

    # register a console writer and connect the writer to the statistics logging system
    with SimpleStatsLoggingSetup(env):
        # reset environment
        obs = env.reset()
        # run interaction loop
        for i in range(15):
            # sample random action
            action = env.action_space.sample()

            # take actual environment step
            obs, reward, done, info = env.step(action)


if __name__ == "__main__":
    """ main """
    main()
