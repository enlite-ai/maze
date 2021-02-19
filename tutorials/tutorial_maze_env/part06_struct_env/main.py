""" Test script CoreEnv """
from tutorials.tutorial_maze_env.part06_struct_env.env.struct_env import struct_env_factory


def main():
    # init maze environment including observation and action interfaces
    struct_env = struct_env_factory(max_pieces_in_inventory=200,
                                    raw_piece_size=(100, 100),
                                    static_demand=[(30, 15)])

    # reset env
    obs_step1 = struct_env.reset()

    print("action_space 1:     ", struct_env.action_space)
    print("observation_space 1:", struct_env.observation_space)
    print("observation 1:      ", obs_step1.keys())

    # take first env step
    action_1 = struct_env.action_space.sample()
    obs_step2, rew, done, info = struct_env.step(action=action_1)

    print("action_space 2:     ", struct_env.action_space)
    print("observation_space 2:", struct_env.observation_space)
    print("observation 2:      ", obs_step2.keys())

    # take second env step
    action_2 = struct_env.action_space.sample()
    obs_step1 = struct_env.step(action=action_2)


if __name__ == "__main__":
    """ main """
    main()
