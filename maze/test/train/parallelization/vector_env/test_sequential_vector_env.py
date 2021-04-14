from maze.core.agent.random_policy import DistributedRandomPolicy
from maze.test.shared_test_utils.helper_functions import build_dummy_structured_env
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv


def test_vectorized_rollout():
    concurrency = 3
    vectorized_env = SequentialVectorEnv([build_dummy_structured_env] * concurrency)

    standard_env = build_dummy_structured_env()
    assert vectorized_env.observation_spaces_dict == standard_env.observation_spaces_dict
    assert vectorized_env.action_spaces_dict == standard_env.action_spaces_dict

    policy = DistributedRandomPolicy(vectorized_env.action_spaces_dict, concurrency=concurrency)

    observation = vectorized_env.reset()
    for _ in range(3):
        action = policy.compute_action(observation, actor_id=vectorized_env.actor_id(), maze_state=None)
        observation, reward, done, info = vectorized_env.step(action)
