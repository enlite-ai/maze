"""Multi-step A2C implementation."""
from collections import defaultdict
from typing import Union, Optional

import torch
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.perception.perception_utils import convert_to_torch
from maze.train.parallelization.distributed_env.distributed_env import BaseDistributedEnv
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.train.trainers.common.actor_critic.actor_critic_trainer import MultiStepActorCritic
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.ppo.ppo_algorithm_config import PPOAlgorithmConfig


class MultiStepA2C(MultiStepActorCritic):
    """Multi step advantage actor critic.

    :param algorithm_config: Algorithm parameters.
    :param env: Distributed structured environment
    :param eval_env: Evaluation distributed structured environment
    :param model: Structured torch actor critic model.
    :param initial_state: path to initial state (policy weights, critic weights, optimizer state)
    :param model_selection: Optional model selection class, receives model evaluation results.
    """

    def __init__(self,
                 algorithm_config: A2CAlgorithmConfig,
                 env: Union[BaseDistributedEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv],
                 eval_env: [BaseDistributedEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv],
                 model: TorchActorCritic,
                 model_selection: Optional[BestModelSelection],
                 initial_state: Optional[str] = None):
        super().__init__(algorithm_config=algorithm_config, env=env, eval_env=eval_env, model=model,
                         model_selection=model_selection, initial_state=initial_state)

    @override(MultiStepActorCritic)
    def _update(self) -> None:
        """Perform policy update.
        """

        # collect observations
        obs, rews, dones, actions = self._rollout()

        # convert observations to tensors
        obs_t = convert_to_torch(obs, device=self.algorithm_config.device, cast=None, in_place=False)

        # compute bootstrapped returns
        returns, values, detached_values = \
            self.model.critic.bootstrap_returns(obs_t, rews, dones, gamma=self.algorithm_config.gamma,
                                                gae_lambda=self.algorithm_config.gae_lambda)

        # compute action log-probabilities of actions taken
        action_log_probs, step_action_dist = self._action_log_probs_and_dists(obs_t, actions)

        # compute entropies
        entropies = [action_dist.entropy().mean() for action_dist in step_action_dist.values()]

        # compute advantages
        advantages = [returns[step_id] - detached_values[step_id]
                      for step_id in self.step_action_keys]

        # normalize advantages
        advantages = self._normalize_advantages(advantages)

        # compute value loss
        value_losses = []
        for step_id in range(self.model.critic.num_critics):
            value_loss = (returns[step_id] - values[step_id]).pow(2).mean()
            value_losses.append(value_loss)
        value_loss = sum(value_losses)

        # compute policy loss
        policy_losses = []
        for step_id in self.step_action_keys:

            # accumulate independent action losses
            step_policy_loss = torch.tensor(0.0).to(self.algorithm_config.device)
            for key in self.step_action_keys[step_id]:

                # prepare advantages
                action_advantages = advantages[step_id].detach()
                while action_advantages.ndim < action_log_probs[step_id][key].ndimension():
                    action_advantages = action_advantages.unsqueeze(dim=-1)

                # compute policy gradient objective
                step_policy_loss -= (action_advantages * action_log_probs[step_id][key]).mean()

            policy_losses.append(step_policy_loss)

        # perform gradient step
        self._gradient_step(policy_losses=policy_losses, entropies=entropies, value_loss=value_loss)

        # collect training stats for logging
        policy_train_stats = [defaultdict(list) for _ in range(self.num_env_sub_steps)]
        critic_train_stats = [defaultdict(list) for _ in range(self.model.critic.num_critics)]
        self._append_train_stats(policy_train_stats, critic_train_stats,
                                 policy_losses, entropies, detached_values, value_losses)

        # fire logging events
        self._log_train_stats(policy_train_stats, critic_train_stats)
