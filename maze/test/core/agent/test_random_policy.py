from maze.core.agent.random_policy import RandomPolicy
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


def test_default_action_space_sampling():
    env = build_dummy_maze_env()
    policy = RandomPolicy(env.action_spaces_dict)
    action = policy.compute_action(observation=env.observation_space.sample(), maze_state=None)
    assert action in env.action_space
