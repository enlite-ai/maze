from maze.core.agent.random_policy import DistributedRandomPolicy
from maze.test.shared_test_utils.helper_functions import build_dummy_structured_env
from maze.train.parallelization.distributed_env.subproc_distributed_env import SubprocStructuredDistributedEnv


def test_distributed_rollout():
    concurrency = 3
    distributed_env = SubprocStructuredDistributedEnv([build_dummy_structured_env] * concurrency)

    standard_env = build_dummy_structured_env()
    assert distributed_env.observation_spaces_dict == standard_env.observation_spaces_dict
    assert distributed_env.action_spaces_dict == standard_env.action_spaces_dict

    policy = DistributedRandomPolicy(distributed_env.action_spaces_dict, concurrency=concurrency)

    observation = distributed_env.reset()
    for _ in range(3):
        policy_id, _ = distributed_env.actor_id()
        action = policy.compute_action(observation, policy_id=policy_id, maze_state=None)
        observation, reward, done, info = distributed_env.step(action)
