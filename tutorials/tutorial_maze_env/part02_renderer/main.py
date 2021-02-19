""" Test script CoreEnv """
from tutorials.tutorial_maze_env.part02_renderer.env.core_env import Cutting2DCoreEnvironment
from tutorials.tutorial_maze_env.part02_renderer.env.maze_action import Cutting2DMazeAction


def main():
    # init and reset core environment
    core_env = Cutting2DCoreEnvironment(max_pieces_in_inventory=200, raw_piece_size=[100, 100],
                                        static_demand=(30, 15))
    maze_state = core_env.reset()
    # run interaction loop
    for i in range(15):
        # create cutting maze_action
        maze_action = Cutting2DMazeAction(piece_id=0, rotate=False, reverse_cutting_order=False)

        # render current state along with next maze_action
        core_env.renderer.render(maze_state, maze_action, None)

        # take actual environment step
        maze_state, reward, done, info = core_env.step(maze_action)
        print(f"reward {reward} | done {done} | info {info}")


if __name__ == "__main__":
    """ main """
    main()
