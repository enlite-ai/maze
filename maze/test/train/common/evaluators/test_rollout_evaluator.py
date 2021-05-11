from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.log_stats.log_stats import increment_log_step, LogStatsLevel
from maze.core.wrappers.maze_gym_env_wrapper import make_gym_maze_env
from maze.core.wrappers.time_limit_wrapper import TimeLimitWrapper
from maze.test.shared_test_utils.helper_functions import flatten_concat_probabilistic_policy_for_env
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.trainers.common.model_selection.model_selection_base import ModelSelectionBase


class _MockModelSelection(ModelSelectionBase):
    def __init__(self):
        self.update_count = 0

    def update(self, reward: float) -> None:
        """Count the updates"""
        self.update_count += 1


def test_rollout_evaluator():
    env = SequentialVectorEnv([lambda: make_gym_maze_env("CartPole-v0")] * 2)
    policy = flatten_concat_probabilistic_policy_for_env(make_gym_maze_env("CartPole-v0"))
    model_selection = _MockModelSelection()

    evaluator = RolloutEvaluator(eval_env=env, n_episodes=3, model_selection=model_selection)
    for i in range(2):
        evaluator.evaluate(policy)
        increment_log_step()

    assert model_selection.update_count == 2
    assert evaluator.eval_env.get_stats_value(
        BaseEnvEvents.reward,
        LogStatsLevel.EPOCH,
        name="total_episode_count"
    ) >= 2 * 3


def test_does_not_carry_over_stats_from_unfinished_episodes():
    policy = flatten_concat_probabilistic_policy_for_env(make_gym_maze_env("CartPole-v0"))

    # Wrap envs in a time-limit wrapper
    env = SequentialVectorEnv([lambda: TimeLimitWrapper.wrap(make_gym_maze_env("CartPole-v0"))] * 2)

    # Make one env slower than the other
    env.envs[0].max_episode_steps = 2
    env.envs[1].max_episode_steps = 10

    evaluator = RolloutEvaluator(eval_env=env, n_episodes=1, model_selection=None)
    for i in range(2):
        evaluator.evaluate(policy)
        increment_log_step()

        # We should get just one episode counted in stats
        assert evaluator.eval_env.get_stats_value(
            BaseEnvEvents.reward,
            LogStatsLevel.EPOCH,
            name="episode_count"
        ) == 1
