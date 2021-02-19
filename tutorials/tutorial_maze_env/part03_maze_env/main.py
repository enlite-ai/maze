""" Test script CoreEnv """
from tutorials.tutorial_maze_env.part03_maze_env.env.maze_env import maze_env_factory


def main():
    # init maze environment including observation and action interfaces
    env = maze_env_factory(max_pieces_in_inventory=10,
                           raw_piece_size=[100, 100],
                           static_demand=(30, 15))

    # reset environment
    obs = env.reset()
    # run interaction loop
    for i in range(15):
        # sample random action
        action = env.action_space.sample()

        # take actual environment step
        obs, reward, done, info = env.step(action)
        print(f"reward {reward} | done {done} | info {info}")


if __name__ == "__main__":
    """ main """
    main()
