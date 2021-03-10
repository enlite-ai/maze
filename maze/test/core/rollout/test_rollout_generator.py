from typing import Dict

from maze.core.agent.random_policy import RandomPolicy, DistributedRandomPolicy
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.test.shared_test_utils.helper_functions import build_dummy_structured_env
from maze.test.shared_test_utils.helper_functions import flatten_concat_probabilistic_policy_for_env
from maze.train.parallelization.distributed_env.dummy_distributed_env import DummyStructuredDistributedEnv


def test_standard_rollout():
    env = build_dummy_structured_env()
    rollout_generator = RolloutGenerator(env=env)
    policy = RandomPolicy(env.action_spaces_dict)
    trajectory = rollout_generator.rollout(policy, n_steps=10)

    assert len(trajectory) == 10

    sub_step_keys = env.action_spaces_dict.keys()
    for record in trajectory.step_records:
        assert sub_step_keys == record.actions.keys()
        assert sub_step_keys == record.observations.keys()
        assert sub_step_keys == record.rewards.keys()

        assert record.batch_shape == None
        for step_key in sub_step_keys:
            assert record.observations[step_key] in env.observation_spaces_dict[step_key]
            assert record.actions[step_key] in env.action_spaces_dict[step_key]


def test_distributed_rollout():
    concurrency = 3
    env = DummyStructuredDistributedEnv([build_dummy_structured_env] * concurrency)
    rollout_generator = RolloutGenerator(env=env)
    policy = DistributedRandomPolicy(env.action_spaces_dict, concurrency=concurrency)
    trajectory = rollout_generator.rollout(policy, n_steps=10)

    assert len(trajectory) == 10

    sub_step_keys = env.action_spaces_dict.keys()
    for record in trajectory.step_records:
        assert sub_step_keys == record.actions.keys()
        assert sub_step_keys == record.observations.keys()
        assert sub_step_keys == record.rewards.keys()

        assert record.batch_shape == [concurrency]
        # The first dimension of the observations should correspond to the distributed env concurrency
        # (We just check the very first array present in the first observation)
        first_sub_step_obs: Dict = list(record.observations.values())[0]
        first_obs_value = list(first_sub_step_obs.values())[0]
        assert first_obs_value.shape[0] == concurrency


def test_standard_rollout_with_logits_and_stats():
    env = build_dummy_structured_env()
    rollout_generator = RolloutGenerator(env=env, record_stats=True, record_logits=True)
    policy = flatten_concat_probabilistic_policy_for_env(env)  # We need a torch policy to be able to record logits
    trajectory = rollout_generator.rollout(policy, n_steps=10)

    assert len(trajectory) == 10

    sub_step_keys = env.action_spaces_dict.keys()
    for record in trajectory.step_records:
        assert record.stats is not None
        assert sub_step_keys == record.logits.keys()

        for step_key in sub_step_keys:
            assert record.logits[step_key].keys() == record.actions[step_key].keys()
