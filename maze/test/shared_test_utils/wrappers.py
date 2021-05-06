"""Contains testing helper functions for wrappers."""
from typing import List, Callable

import numpy as np
from maze.core.env.maze_env import MazeEnv


def assert_wrapper_clone_from(make_env: Callable[[], MazeEnv], assert_member_list: List[str] = None):
    """ Asserts that the clone_from and step_without_observation functions work properly for wrappers.

    :param make_env: Instantiates a MazeEnv.
    :param assert_member_list: A list of member variables that should be asserted for equality.
    """

    if not assert_member_list:
        assert_member_list = list()

    # init main and cloned env
    main_env = make_env()
    cloned_env = make_env()

    # reset main env and take a random step
    main_env.reset()
    print("\nmain_env.step")
    main_env.step(main_env.action_space.sample())

    # clone state from main env
    cloned_env.clone_from(main_env)

    for member in assert_member_list:
        assert getattr(main_env, member) == getattr(cloned_env, member)

    # take the same action in main and cloned env
    action = main_env.action_space.sample()

    print("\nmain_env.step")
    print(main_env.core_env.context._should_clear_events)
    obs, rew, done, info = main_env.step(action)

    print("\ncloned_env.step")
    print(cloned_env.core_env.context._should_clear_events)
    obs_sim, rew_sim, done_sim, info_sim = cloned_env.step(action)

    # assert that they return the same
    assert rew == rew_sim
    assert done == done_sim
    assert np.all(obs["observation"] == obs_sim["observation"])

    for member in assert_member_list:
        assert getattr(main_env, member) == getattr(cloned_env, member)

    # take the same action in main and cloned env
    action = main_env.action_space.sample()
    rew, done, info = main_env.step_without_observation(action)
    rew_sim, done_sim, info_sim = cloned_env.step_without_observation(action)

    assert rew == rew_sim
    assert done == done_sim

    for member in assert_member_list:
        assert getattr(main_env, member) == getattr(cloned_env, member)
