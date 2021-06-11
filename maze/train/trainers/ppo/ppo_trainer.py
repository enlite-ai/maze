"""Multi-step multi-agent PPO implementation."""
from collections import defaultdict
from typing import Union, Dict, Optional

import numpy as np
import torch

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv, StepKeyType
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
            policy_output_old, critic_output_old = self.model.compute_actor_critic_output(record)
            returns = self.model.critic.compute_structured_return(gamma=self.algorithm_config.gamma,
                                                                  gae_lambda=self.algorithm_config.gae_lambda,
                                                                  rewards=record.rewards,
                                                                  values=critic_output_old.detached_values,
                                                                  dones=record.dones[-1])
            action_log_probs_old = self.model.policy.compute_action_log_probs(policy_output_old,
                                                                              record.actions_dict)
            # manually empty GPU cache
            torch.cuda.empty_cache()

        # flatten items for batch processing/
        returns = {kk: rr.flatten() for kk, rr in returns.items()}
        self._flatten_sub_step_items(record.actions_dict)
        self._flatten_sub_step_items(record.observations_dict)
        self._flatten_sub_step_items(action_log_probs_old)
        critic_output_old.reshape(list(returns.values())[0].shape)

        # iterate ppo optimization epochs
        critic_train_stats = defaultdict(lambda: defaultdict(list))
        policy_train_stats = defaultdict(lambda: defaultdict(list))
        n_samples = self.env.n_envs * self.algorithm_config.n_rollout_steps
        for k in range(self.algorithm_config.n_optimization_epochs):
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

                # Produce policy and critic output
                policy_output, critic_output = self.model.compute_actor_critic_output(batch_record)

                # Compute action log probabilities with the original actions
                action_log_probs = self.model.policy.compute_action_log_probs(policy_output, batch_record.actions_dict)

                # compute advantages
                advantages = {kk: rr[batch_idxs] - critic_output_old.detached_values[kk][batch_idxs]
                              for kk, rr in returns.items()}

                # normalize advantages
                advantages = self._normalize_advantages(advantages)

                # compute value loss
                if self.model.critic.num_critics == 1:
                    key = list(returns.keys())[0]
                    value_losses = {key: (returns[key][batch_idxs] - critic_output.values[key][0]).pow(2).mean()}
                else:
                    value_losses = {step_key: (returns[step_key][batch_idxs] - values).pow(2).mean()
                                    for step_key, values in critic_output.values.items()}

                # compute policy loss
                policy_losses = dict()
                entropies = dict()
                for idx, substep_record in enumerate(batch_record.substep_records):

                    # compute entropies
                    step_entropy = policy_output.entropy[idx].mean()
                    entropies[idx] = step_entropy

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

                    policy_losses[idx] = step_policy_loss

                # perform gradient step
                self._gradient_step(policy_losses=policy_losses, entropies=entropies, value_losses=value_losses)

                # append training stats for logging
                self._append_train_stats(policy_train_stats, critic_train_stats,
                                         policy_losses, entropies, critic_output_old.detached_values, value_losses)

        # fire logging events
        self._log_train_stats(policy_train_stats, critic_train_stats)

    def _flatten_sub_step_items(self, step_items: Dict[StepKeyType, Dict[str, torch.Tensor]]) -> None:
        """Flattens sub-step items for batch processing in PPO.
        :param step_items: Dict of items to be flattened.
        """
        for substep_key, substep_dict in step_items.items():
            for key in substep_dict.keys():
                substep_dict[key] = torch.flatten(substep_dict[key], start_dim=0, end_dim=1)
