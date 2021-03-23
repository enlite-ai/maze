from typing import Dict, Any, Tuple

from maze.core.agent.random_policy import RandomPolicy, DistributedRandomPolicy
from maze.core.env.base_env import BaseEnv
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.wrappers.time_limit_wrapper import TimeLimitWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_structured_env, build_dummy_maze_env
from maze.test.shared_test_utils.helper_functions import flatten_concat_probabilistic_policy_for_env
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv


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
    env = SequentialVectorEnv([build_dummy_structured_env] * concurrency)
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


def test_standard_rollout_with_logits_and_step_stats():
    env = build_dummy_structured_env()
    rollout_generator = RolloutGenerator(env=env, record_step_stats=True, record_logits=True)
    policy = flatten_concat_probabilistic_policy_for_env(env)  # We need a torch policy to be able to record logits
    trajectory = rollout_generator.rollout(policy, n_steps=10)

    assert len(trajectory) == 10

    sub_step_keys = env.action_spaces_dict.keys()
    for record in trajectory.step_records:
        assert record.step_stats is not None
        assert sub_step_keys == record.logits.keys()

        for step_key in sub_step_keys:
            assert record.logits[step_key].keys() == record.actions[step_key].keys()


def test_terminates_on_done():
    env = build_dummy_maze_env()
    env = TimeLimitWrapper.wrap(env, max_episode_steps=5)
    policy = RandomPolicy(env.action_spaces_dict)

    # Normal operation (should reset the env automatically and continue rollout)
    rollout_generator = RolloutGenerator(env=env)
    trajectory = rollout_generator.rollout(policy, n_steps=10)
    assert len(trajectory) == 10

    # Terminate on done
    rollout_generator = RolloutGenerator(env=env, terminate_on_done=True)
    trajectory = rollout_generator.rollout(policy, n_steps=10)
    assert len(trajectory) == 5


def test_handles_done_in_substep_with_recorded_episode_stats():
    class _FiveSubstepsLimitWrapper(TimeLimitWrapper):
        """Returns done after 5 sub-steps (not flat env steps!)"""
        def __init__(self, env: BaseEnv):
            super().__init__(env)
            self.elapsed_sub_steps = 0

        def step(self, action: Any) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
            """Return done after 5 sub-steps"""
            observation, reward, done, info = self.env.step(action)
            self.elapsed_sub_steps += 1
            return observation, reward, done or self.elapsed_sub_steps >= 5, info

        def reset(self) -> Any:
            self.elapsed_sub_steps = 0
            return self.env.reset()

    env = build_dummy_structured_env()
    env = _FiveSubstepsLimitWrapper.wrap(env)
    policy = RandomPolicy(env.action_spaces_dict)

    # -- Normal operation (should reset the env automatically and continue rollout) --
    rollout_generator = RolloutGenerator(env=env, record_episode_stats=True)
    trajectory = rollout_generator.rollout(policy, n_steps=10)
    assert len(trajectory) == 10

    # The done step records should have data for the first sub-step only
    dones = 0
    for step_record in trajectory.step_records:
        if step_record.is_done():
            assert [0] == list(step_record.observations.keys())
            dones += 1
            assert step_record.episode_stats is not None
        else:
            assert [0, 1] == list(step_record.observations.keys())
            assert step_record.episode_stats is None
    assert dones == 3  # Each episode is done after 5 sub-steps, i.e. 3 structured steps get recorded => 3 episodes fit

    # -- Terminate on done --
    rollout_generator = RolloutGenerator(env=env, terminate_on_done=True)
    trajectory = rollout_generator.rollout(policy, n_steps=10)
    assert len(trajectory) == 3
    assert trajectory.is_done()
    assert [0] == list(trajectory.step_records[-1].observations.keys())
