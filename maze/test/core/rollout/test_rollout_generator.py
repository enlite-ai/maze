from typing import Dict

from maze.core.agent.random_policy import DistributedRandomPolicy
from maze.core.rollout.rollout_generator import DistributedEnvRolloutGenerator
from maze.test.shared_test_utils.helper_functions import build_dummy_structured_env
from maze.train.parallelization.distributed_env.dummy_distributed_env import DummyStructuredDistributedEnv


def test_distributed_rollout():
    concurrency = 3
    env = DummyStructuredDistributedEnv([build_dummy_structured_env] * concurrency)
    rollout_generator = DistributedEnvRolloutGenerator(env=env)
    policy = DistributedRandomPolicy(env.action_spaces_dict, concurrency=concurrency)
    records = rollout_generator.rollout(policy, n_steps=10)

    assert len(records) == 10

    sub_step_keys = env.action_spaces_dict.keys()
    for record in records:
        assert sub_step_keys == record.actions.keys()
        assert sub_step_keys == record.observations.keys()
        assert sub_step_keys == record.rewards.keys()

        assert record.batch_shape == [concurrency]
        # The first dimension of the observations should correspond to the distributed env concurrency
        # (We just check the very first array present in the first observation)
        first_sub_step_obs: Dict = list(record.observations.values())[0]
        first_obs_value = list(first_sub_step_obs.values())[0]
        assert first_obs_value.shape[0] == concurrency
