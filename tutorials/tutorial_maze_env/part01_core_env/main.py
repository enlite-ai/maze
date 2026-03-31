"""Test script CoreEnv"""

from __future__ import annotations

from tutorial_maze_env.part01_core_env.env.core_env import Cutting2DCoreEnvironment
from tutorial_maze_env.part01_core_env.env.maze_action import Cutting2DMazeAction


def main():
    # init and reset core environment
    core_env = Cutting2DCoreEnvironment(max_pieces_in_inventory=200, raw_piece_size=[100, 100], static_demand=(30, 15))

    maze_state, _ = core_env.reset()
    # run interaction loop
    for _ in range(15):
        # create cutting maze_action
        maze_action = Cutting2DMazeAction(piece_id=0, rotate=False, reverse_cutting_order=False)
        # take actual environment step
        maze_state, reward, terminated, truncated, info = core_env.step(maze_action)
        print(f'reward {reward} | terminated {terminated} | truncated {truncated} | info {info}')


if __name__ == '__main__':
    """ main """
    main()
