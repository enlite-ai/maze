"""Torch policy mechanics tests."""
import pytest

from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env, \
    flatten_concat_probabilistic_policy_for_env, build_dummy_maze_environment_with_discrete_action_space


def test_actor_id_is_optional_for_single_network_policies():
    env = build_dummy_maze_env()
    policy = flatten_concat_probabilistic_policy_for_env(env)

    obs = env.reset()
    action = policy.compute_action(obs)  # No actor ID provided
    assert all([key in env.action_space.spaces for key in action.keys()])


def test_torch_policy_with_discrete_only_action_space():
    env = build_dummy_maze_environment_with_discrete_action_space()
    policy = flatten_concat_probabilistic_policy_for_env(env)

    obs = env.reset()
    action = policy.compute_action(obs)  # No actor ID provided
    assert all([key in env.action_space.spaces for key in action.keys()])

    top_actions, probs = policy.compute_top_action_candidates(obs, num_candidates=5, maze_state=None, env=None)

    # Check number returned
    assert len(top_actions) == 5 == len(probs)

    # Probabilities are sorted
    assert list(sorted(probs, reverse=True)) == probs

    for action in top_actions:
        assert all([key in env.action_space.spaces for key in action.keys()])
