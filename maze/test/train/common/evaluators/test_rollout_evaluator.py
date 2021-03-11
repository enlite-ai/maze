from torch import nn

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.maze_env import MazeEnv
from maze.core.log_stats.log_stats import increment_log_step, LogStatsLevel
from maze.core.wrappers.maze_gym_env_wrapper import make_gym_maze_env
from maze.perception.models.built_in.flatten_concat import FlattenConcatPolicyNet
from maze.perception.models.custom_model_composer import CustomModelComposer
from maze.perception.models.policies import ProbabilisticPolicyComposer
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.parallelization.distributed_env.dummy_distributed_env import DummyStructuredDistributedEnv
from maze.train.trainers.common.model_selection.model_selection_base import ModelSelectionBase


def flatten_concat_policy_for_env(env: MazeEnv):
    """Build a policy using a small flatten-concat network for a given env."""
    composer = CustomModelComposer(
        action_spaces_dict=env.action_spaces_dict,
        observation_spaces_dict=env.observation_spaces_dict,
        distribution_mapper_config={},
        policy=dict(
            _target_=ProbabilisticPolicyComposer,
            networks=[dict(_target_=FlattenConcatPolicyNet, non_lin=nn.Tanh, hidden_units=[32, 32])]
        ),
        critic=None
    )
    return TorchPolicy(
        networks=composer.policy.networks,
        distribution_mapper=composer.distribution_mapper,
        device="cpu")


class _MockModelSelection(ModelSelectionBase):
    def __init__(self):
        self.update_count = 0

    def update(self, reward: float) -> None:
        """Count the updates"""
        self.update_count += 1


def test_rollout_evaluator():
    env = DummyStructuredDistributedEnv([lambda: make_gym_maze_env("CartPole-v0")] * 2)
    policy = flatten_concat_policy_for_env(make_gym_maze_env("CartPole-v0"))
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
