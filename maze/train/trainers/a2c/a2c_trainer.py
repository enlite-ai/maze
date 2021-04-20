"""Multi-step multi-agent A2C implementation."""
from collections import defaultdict
from typing import Union, Optional

import torch

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.train.parallelization.vector_env.vector_env import VectorEnv
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.train.trainers.common.actor_critic.actor_critic_trainer import ActorCritic
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection


class A2C(ActorCritic):
    """Advantage Actor Critic. Suitable for multi-step and multi-agent scenarios.

    :param algorithm_config: Algorithm parameters.
    :param env: Distributed structured environment
    :param eval_env: Evaluation distributed structured environment
    :param model: Structured torch actor critic model.
    :param initial_state: path to initial state (policy weights, critic weights, optimizer state)
    :param model_selection: Optional model selection class, receives model evaluation results.
    """

    def __init__(self,
                 algorithm_config: A2CAlgorithmConfig,
                 env: Union[VectorEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv],
                 eval_env: Optional[Union[VectorEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
                 model: TorchActorCritic,
                 model_selection: Optional[BestModelSelection],
                 initial_state: Optional[str] = None):
        super().__init__(algorithm_config=algorithm_config, env=env, eval_env=eval_env, model=model,
                         model_selection=model_selection, initial_state=initial_state)

    @override(ActorCritic)
    def _update(self) -> None:
        """Perform policy update.
        """
        # collect observations
        record = self._rollout()

        # compute bootstrapped returns
        returns, values, detached_values = self.model.critic.bootstrap_returns(
            record=record,
            gamma=self.algorithm_config.gamma,
            gae_lambda=self.algorithm_config.gae_lambda)

        # compute action log-probabilities of actions taken
        action_log_probs, step_action_dist = self._action_log_probs_and_dists(record)

        # compute entropies
        entropies = [action_dist.entropy().mean() for action_dist in step_action_dist]

        # compute advantages
        advantages = [ret - detached_val for ret, detached_val in zip(returns, detached_values)]

        # normalize advantages
        advantages = self._normalize_advantages(advantages)

        # compute value loss
        if self.model.critic.num_critics == 1:
            value_losses = [(returns[0] - values[0]).pow(2).mean()]
        else:
            value_losses = [(ret - val).pow(2).mean() for ret, val in zip(returns, values)]
        value_loss = sum(value_losses)

        # compute policy loss, iterating across all sub-steps
        policy_losses = []
        for advantage, action_log_probs_dict in zip(advantages, action_log_probs):

            # accumulate independent action losses, iterating components of the action
            step_policy_loss = torch.tensor(0.0).to(self.algorithm_config.device)
            for action_log_prob in action_log_probs_dict.values():
                # prepare advantages
                action_advantages = advantage.detach()
                while action_advantages.ndim < action_log_prob.ndimension():
                    action_advantages = action_advantages.unsqueeze(dim=-1)

                # compute policy gradient objective
                step_policy_loss -= (action_advantages * action_log_prob).mean()

            policy_losses.append(step_policy_loss)

        # perform gradient step
        self._gradient_step(policy_losses=policy_losses, entropies=entropies, value_loss=value_loss)

        # collect training stats for logging
        policy_train_stats = defaultdict(lambda: defaultdict(list))
        critic_train_stats = defaultdict(lambda: defaultdict(list))
        self._append_train_stats(policy_train_stats, critic_train_stats,
                                 record.actor_ids, policy_losses, entropies, detached_values, value_losses)

        # fire logging events
        self._log_train_stats(policy_train_stats, critic_train_stats)
