"""Test script CoreEnv"""

from __future__ import annotations

from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.utils.log_stats_utils import SimpleStatsLoggingSetup

from tutorial_maze_env.part05_reward.env.maze_env import maze_env_factory


def main():
    # init maze environment including observation and action interfaces
    env = maze_env_factory(max_pieces_in_inventory=200, raw_piece_size=[100, 100], static_demand=(30, 15))

    # wrap environment with logging wrapper
    env = LogStatsWrapper(env, logging_prefix='main')

    # register a console writer and connect the writer to the statistics logging system
    with SimpleStatsLoggingSetup(env):
        # reset environment
        obs, _ = env.reset()
        # run interaction loop
        for _ in range(15):
            # sample random action
            action = env.action_space.sample()

            # take actual environment step
            maze_state, reward, terminated, truncated, info = env.step(action)
            print(f'reward {reward} | terminated {terminated} | truncated {truncated}| info {info}')


if __name__ == '__main__':
    """ main """
    main()
