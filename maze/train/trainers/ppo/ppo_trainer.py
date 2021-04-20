"""Multi-step multi-agent PPO implementation."""
from collections import defaultdict
from typing import Union, Dict, Optional, List

import numpy as np
import torch

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.trajectory_recording.records.spaces_record import SpacesRecord
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.train.parallelization.vector_env.vector_env import VectorEnv
from maze.train.trainers.common.actor_critic.actor_critic_trainer import ActorCritic
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.ppo.ppo_algorithm_config import PPOAlgorithmConfig


class PPO(ActorCritic):
    """Proximal Policy Optimization trainer. Suitable for multi-step and multi-agent scenarios.

    :param algorithm_config: Algorithm parameters.
    :param env: Distributed structured environment
    :param eval_env: Evaluation distributed structured environment
    :param model: Structured torch actor critic model.
    :param initial_state: path to initial state (policy weights, critic weights, optimizer state)
    :param model_selection: Optional model selection class, receives model evaluation results.
    """

    def __init__(self,
                 algorithm_config: PPOAlgorithmConfig,
                 env: Union[VectorEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv],
                 eval_env: Optional[Union[VectorEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
                 model: TorchActorCritic,
                 model_selection: Optional[BestModelSelection],
                 initial_state: Optional[str] = None):
        super().__init__(algorithm_config=algorithm_config, env=env, eval_env=eval_env, model=model,
                         model_selection=model_selection, initial_state=initial_state)

    @override(ActorCritic)
    def _update(self) -> None:
        """Perform ppo policy update.
        """

        # collect observations
        record = self._rollout()

        # compute action log-probabilities of actions taken (aka old action log probs)
        with torch.no_grad():
            action_log_probs_old, _ = self._action_log_probs_and_dists(record)
            # manually empty GPU cache
            torch.cuda.empty_cache()

        # flatten items for batch processing
        self._flatten_sub_step_items(record.actions)
        self._flatten_sub_step_items(record.observations)
        self._flatten_sub_step_items(action_log_probs_old)

        # iterate ppo optimization epochs
        critic_train_stats = defaultdict(lambda: defaultdict(list))
        policy_train_stats = defaultdict(lambda: defaultdict(list))
        n_samples = self.env.n_envs * self.algorithm_config.n_rollout_steps
        for k in range(self.algorithm_config.n_optimization_epochs):

            # compute bootstrapped returns
            with torch.no_grad():
                returns, _, detached_values = self.model.critic.bootstrap_returns(
                    record=record,
                    gamma=self.algorithm_config.gamma,
                    gae_lambda=self.algorithm_config.gae_lambda)
                # manually empty GPU cache
                torch.cuda.empty_cache()

            # convert everything to batch dimension
            returns = [r.flatten() for r in returns]
            detached_values = [dv.flatten() for dv in detached_values]

            # iterate mini-batch updates
            indices = np.random.permutation(n_samples)
            n_batches = n_samples // self.algorithm_config.batch_size
            for i_batch in range(n_batches):
                # manually empty GPU cache
                torch.cuda.empty_cache()

                # sample batch indices
                i0 = i_batch * self.algorithm_config.batch_size
                i1 = i0 + self.algorithm_config.batch_size
                batch_idxs = indices[i0:i1]

                # get batch data into a new spaces record
                batch_record = StructuredSpacesRecord()
                for substep_record in record.substep_records:
                    batch_substep_record = SpacesRecord(
                        actor_id=substep_record.actor_id,
                        action={},
                        observation={}
                    )

                    # observations
                    for key, value in substep_record.observation.items():
                        batch_substep_record.observation[key] = value[batch_idxs]

                    # actions
                    for key, value in substep_record.action.items():
                        batch_substep_record.action[key] = value[batch_idxs]

                    batch_record.append(batch_substep_record)

                # predict values
                values, _ = self.model.critic.predict_values(batch_record)

                # compute advantages
                advantages = [r[batch_idxs] - dv[batch_idxs] for r, dv in zip(returns, detached_values)]

                # normalize advantages
                advantages = self._normalize_advantages(advantages)

                # compute value loss
                if self.model.critic.num_critics == 1:
                    value_losses = [(returns[0][batch_idxs] - values[0]).pow(2).mean()]
                else:
                    value_losses = [(ret[batch_idxs] - val).pow(2).mean() for ret, val in zip(returns, values)]
                value_loss = sum(value_losses)

                # compute log probabilities
                action_log_probs, prob_dists = self._action_log_probs_and_dists(batch_record)

                # compute policy loss
                policy_losses = []
                entropies = []
                for idx, substep_record in enumerate(batch_record.substep_records):

                    # compute entropies
                    step_entropy = prob_dists[idx].entropy().mean()
                    entropies.append(step_entropy)

                    # accumulate independent action losses
                    step_policy_loss = torch.tensor(0.0).to(self.algorithm_config.device)
                    for key in substep_record.action.keys():

                        # get relevant log probs
                        log_probs = action_log_probs[idx][key]
                        old_log_probs = action_log_probs_old[idx][key][batch_idxs]

                        # prepare advantages
                        action_advantages = advantages[idx].detach()
                        while action_advantages.ndim < action_log_probs[idx][key].ndimension():
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
                                         record.actor_ids, policy_losses, entropies, detached_values, value_losses)

        # fire logging events
        self._log_train_stats(policy_train_stats, critic_train_stats)

    def _flatten_sub_step_items(self, step_items: List[Dict[str, torch.Tensor]]) -> None:
        """Flattens sub-step items for batch processing in PPO.
        :param step_items: Dict of items to be flattened.
        """

        for substep_dict in step_items:
            for key in substep_dict.keys():
                substep_dict[key] = torch.flatten(substep_dict[key], start_dim=0, end_dim=1)
