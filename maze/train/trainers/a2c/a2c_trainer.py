"""Multi-step multi-agent A2C implementation."""
from collections import defaultdict

import torch

from maze.core.annotations import override
from maze.train.trainers.common.actor_critic.actor_critic_trainer import ActorCritic


class A2C(ActorCritic):
    """Advantage Actor Critic. Suitable for multi-step and multi-agent scenarios. """

    @override(ActorCritic)
    def _update(self) -> None:
        """Perform policy update.
        """
        # collect observations
        record = self._rollout()

        # compute action log-probabilities and values
        policy_output, critic_output = self.model.compute_actor_critic_output(record)

        # compute returns
        returns = self.model.critic.compute_structured_return(gamma=self.algorithm_config.gamma,
                                                              gae_lambda=self.algorithm_config.gae_lambda,
                                                              rewards=record.rewards,
                                                              values=critic_output.detached_values,
                                                              dones=record.dones[-1])

        # compute entropies
        entropies = [entropy.mean() for entropy in policy_output.entropies]

        # compute advantages
        advantages = [ret - detached_val for ret, detached_val in
                      zip(returns, critic_output.detached_values)]

        # normalize advantages
        advantages = self._normalize_advantages(advantages)

        # compute value loss
        if self.model.critic.num_critics == 1:
            value_losses = [(returns[0] - critic_output.values[0]).pow(2).mean()]
        else:
            value_losses = [(ret - val).pow(2).mean() for ret, val in zip(returns, critic_output.values)]

        # compute policy loss, iterating across all sub-steps
        action_log_probs = policy_output.log_probs_for_actions(record.actions)
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
        self._gradient_step(policy_losses=policy_losses, entropies=entropies, value_losses=value_losses)

        # collect training stats for logging
        policy_train_stats = defaultdict(lambda: defaultdict(list))
        critic_train_stats = defaultdict(lambda: defaultdict(list))
        self._append_train_stats(policy_train_stats, critic_train_stats,
                                 record.actor_ids, policy_losses, entropies,
                                 critic_output.detached_values, value_losses)

        # fire logging events
        self._log_train_stats(policy_train_stats, critic_train_stats)
