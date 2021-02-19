"""Multi-step PPO implementation."""
from collections import defaultdict
from typing import Union, Dict, Optional

import numpy as np
import torch
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.perception.perception_utils import convert_to_torch
from maze.train.parallelization.distributed_env.distributed_env import BaseDistributedEnv
from maze.train.trainers.common.actor_critic.actor_critic_trainer import MultiStepActorCritic
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.ppo.ppo_algorithm_config import PPOAlgorithmConfig


class MultiStepPPO(MultiStepActorCritic):
    """Multi step Proximal Policy Optimization.

    :param algorithm_config: Algorithm parameters.
    :param env: Distributed structured environment
    :param eval_env: Evaluation distributed structured environment
    :param model: Structured torch actor critic model.
    :param initial_state: path to initial state (policy weights, critic weights, optimizer state)
    :param model_selection: Optional model selection class, receives model evaluation results.
    """

    def __init__(self,
                 algorithm_config: PPOAlgorithmConfig,
                 env: Union[BaseDistributedEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv],
                 eval_env: [BaseDistributedEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv],
                 model: TorchActorCritic,
                 model_selection: Optional[BestModelSelection],
                 initial_state: Optional[str] = None):
        super().__init__(algorithm_config=algorithm_config, env=env, eval_env=eval_env, model=model,
                         model_selection=model_selection, initial_state=initial_state)

    @override(MultiStepActorCritic)
    def _update(self) -> None:
        """Perform ppo policy update.
        """

        # collect observations
        obs, rews, dones, actions = self._rollout()

        # convert observations to tensors
        obs_t = convert_to_torch(obs, device=self.algorithm_config.device, cast=None, in_place=False)

        # compute action log-probabilities of actions taken (aka old action log probs)
        with torch.no_grad():
            action_log_probs_old, _ = self._action_log_probs_and_dists(obs_t, actions)

        # flatten items for batch processing
        actions = self._flatten_sub_step_items(actions)
        obs_t = self._flatten_sub_step_items(obs_t)
        action_log_probs_old = self._flatten_sub_step_items(action_log_probs_old)

        # iterate ppo optimization epochs
        critic_train_stats = [defaultdict(list) for _ in range(self.model.critic.num_critics)]
        policy_train_stats = [defaultdict(list) for _ in range(self.num_env_sub_steps)]
        n_samples = self.env.num_envs * self.algorithm_config.n_rollout_steps
        for k in range(self.algorithm_config.n_optimization_epochs):

            # compute bootstrapped returns
            returns, _, detached_values = self.model.critic.bootstrap_returns(
                obs_t, rews, dones,
                gamma=self.algorithm_config.gamma,
                gae_lambda=self.algorithm_config.gae_lambda)

            # convert everything to batch dimension
            for step_id in self.sub_step_keys:
                returns[step_id] = returns[step_id].flatten()
                detached_values[step_id] = detached_values[step_id].flatten()

            # iterate mini-batch updates
            indices = np.random.permutation(n_samples)
            n_batches = n_samples // self.algorithm_config.batch_size
            for i_batch in range(n_batches):

                # sample batch indices
                i0 = i_batch * self.algorithm_config.batch_size
                i1 = i0 + self.algorithm_config.batch_size
                batch_idxs = indices[i0:i1]

                # get batch data
                batch_obs = defaultdict(dict)
                batch_actions = defaultdict(dict)
                for step_id in self.step_action_keys:

                    # observations
                    for key in obs_t[step_id].keys():
                        batch_obs[step_id][key] = obs_t[step_id][key][batch_idxs]

                    # actions
                    for key in actions[step_id].keys():
                        batch_actions[step_id][key] = actions[step_id][key][batch_idxs]

                # predict values (the observation list transformed inline to structured env dict style)
                values, _ = self.model.critic.predict_values(batch_obs)

                # compute advantages
                advantages = [returns[step_id][batch_idxs] - detached_values[step_id][batch_idxs]
                              for step_id in self.step_action_keys]

                # normalize advantages
                advantages = self._normalize_advantages(advantages)

                # compute value loss
                value_losses = []
                for critic_idx in range(self.model.critic.num_critics):
                    value_loss = (returns[critic_idx][batch_idxs] - values[critic_idx]).pow(2).mean()
                    value_losses.append(value_loss)
                value_loss = sum(value_losses)

                # compute log probabilities
                action_log_probs, prob_dists = self._action_log_probs_and_dists(batch_obs, batch_actions)

                # compute policy loss
                policy_losses = []
                entropies = []
                for step_id in self.step_action_keys:

                    # compute entropies
                    step_entropy = prob_dists[step_id].entropy().mean()
                    entropies.append(step_entropy)

                    # accumulate independent action losses
                    step_policy_loss = torch.tensor(0.0).to(self.algorithm_config.device)
                    for key in self.step_action_keys[step_id]:

                        # get relevant log probs
                        log_probs = action_log_probs[step_id][key]
                        old_log_probs = action_log_probs_old[step_id][key][batch_idxs]

                        # prepare advantages
                        action_advantages = advantages[step_id].detach()
                        while action_advantages.ndim < action_log_probs[step_id][key].ndimension():
                            action_advantages = action_advantages.unsqueeze(dim=-1)

                        # compute surrogate objective
                        ratio = torch.exp(log_probs - old_log_probs)
                        surr1 = ratio * action_advantages
                        surr2 = torch.clamp(ratio,
                                            1.0 - self.algorithm_config.clip_range,
                                            1.0 + self.algorithm_config.clip_range) * action_advantages
                        action_loss = -torch.min(surr1, surr2).mean()
                        step_policy_loss += action_loss

                    policy_losses.append(step_policy_loss)

                # perform gradient step
                self._gradient_step(policy_losses=policy_losses, entropies=entropies, value_loss=value_loss)

                # append training stats for logging
                self._append_train_stats(policy_train_stats, critic_train_stats,
                                         policy_losses, entropies, detached_values, value_losses)

        # fire logging events
        self._log_train_stats(policy_train_stats, critic_train_stats)

    def _flatten_sub_step_items(self, step_items: Dict[Union[str, int], Dict[str, torch.Tensor]]) \
            -> Dict[Union[str, int], Dict[str, torch.Tensor]]:
        """Flattens sub-step items for batch processing in PPO.
        :param step_items: Dict of items to be flattened.
        :return: Dict of flattened items.
        """

        # iterate sub-steps
        for step_id in self.step_action_keys:

            step_item = step_items[step_id]
            for key in step_item.keys():
                step_items[step_id][key] = torch.flatten(step_item[key], start_dim=0, end_dim=1)

        return step_items
