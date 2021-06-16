"""Multi-step multi-agent PPO implementation."""
import copy
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from maze.core.annotations import override
from maze.core.trajectory_recording.records.spaces_record import SpacesRecord
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.train.trainers.common.actor_critic.actor_critic_trainer import ActorCritic


class PPO(ActorCritic):
    """Proximal Policy Optimization trainer. Suitable for multi-step and multi-agent scenarios. """

    @override(ActorCritic)
    def _update(self) -> None:
        """Perform ppo policy update.
        """

        # collect observations
        record = self._rollout()

        # iterate ppo optimization epochs
        critic_train_stats = defaultdict(lambda: defaultdict(list))
        policy_train_stats = defaultdict(lambda: defaultdict(list))
        n_samples = self.rollout_generator.env.n_envs * self.algorithm_config.n_rollout_steps

        flat_record = copy.deepcopy(record)
        self._flatten_sub_step_items(flat_record.actions)
        self._flatten_sub_step_items(flat_record.observations)

        for k in range(self.algorithm_config.n_optimization_epochs):
            # compute action log-probabilities of actions taken (aka old action log probs)
            with torch.no_grad():
                policy_output_old, critic_output_old = self.model.compute_actor_critic_output(record)
                returns = self.model.critic.compute_structured_return(gamma=self.algorithm_config.gamma,
                                                                      gae_lambda=self.algorithm_config.gae_lambda,
                                                                      rewards=record.rewards,
                                                                      values=critic_output_old.detached_values,
                                                                      dones=record.dones[-1])
                action_log_probs_old = policy_output_old.log_probs_for_actions(record.actions)
                # manually empty GPU cache
                torch.cuda.empty_cache()

            # flatten items for batch processing/
            returns = [r.flatten() for r in returns]
            self._flatten_sub_step_items(action_log_probs_old)
            critic_output_old.reshape(returns[0].shape)

            # iterate mini-batch updates
            indices = np.random.permutation(n_samples)
            n_batches = int(np.ceil(float(n_samples) / self.algorithm_config.batch_size))
            for i_batch in range(n_batches):
                # manually empty GPU cache
                torch.cuda.empty_cache()

                # sample batch indices
                i0 = i_batch * self.algorithm_config.batch_size
                i1 = i0 + self.algorithm_config.batch_size
                batch_idxs = indices[i0:i1]

                # get batch data into a new spaces record
                batch_record = StructuredSpacesRecord()
                for substep_record in flat_record.substep_records:
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

                # Produce policy and critic output
                policy_output, critic_output = self.model.compute_actor_critic_output(batch_record)

                # Compute action log probabilities with the original actions
                action_log_probs = policy_output.log_probs_for_actions(batch_record.actions)

                # compute advantages
                advantages = [r[batch_idxs] - dv[batch_idxs] for r, dv in
                              zip(returns, critic_output_old.detached_values)]

                # normalize advantages
                advantages = self._normalize_advantages(advantages)

                # compute value loss
                if self.model.critic.num_critics == 1:
                    value_losses = [(returns[0][batch_idxs] - critic_output.values[0]).pow(2).mean()]
                else:
                    value_losses = [(ret[batch_idxs] - val).pow(2).mean() for ret, val in
                                    zip(returns, critic_output.values)]

                # compute policy loss
                policy_losses = list()
                entropies = list()
                for idx, substep_record in enumerate(batch_record.substep_records):

                    # compute entropies
                    entropies.append(policy_output[idx].entropy.mean())

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
                self._gradient_step(policy_losses=policy_losses, entropies=entropies, value_losses=value_losses)

                # append training stats for logging
                self._append_train_stats(policy_train_stats, critic_train_stats,
                                         record.actor_ids,
                                         policy_losses, entropies, critic_output_old.detached_values, value_losses)

        # fire logging events
        self._log_train_stats(policy_train_stats, critic_train_stats)

    @staticmethod
    def _flatten_sub_step_items(step_items: List[Dict[str, torch.Tensor]]) -> None:
        """Flattens sub-step items for batch processing in PPO.
        :param step_items: Dict of items to be flattened.
        """

        for substep_dict in step_items:
            for key in substep_dict.keys():
                substep_dict[key] = torch.flatten(substep_dict[key], start_dim=0, end_dim=1)
