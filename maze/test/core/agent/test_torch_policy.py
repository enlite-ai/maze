"""Torch policy mechanics tests."""

from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env, \
    flatten_concat_probabilistic_policy_for_env


def test_actor_id_is_optional_for_single_network_policies():
    env = build_dummy_maze_env()
    policy = flatten_concat_probabilistic_policy_for_env(env)

    obs = env.reset()
    action = policy.compute_action(obs)  # No actor ID provided
    assert all([key in env.action_space.spaces for key in action.keys()])
