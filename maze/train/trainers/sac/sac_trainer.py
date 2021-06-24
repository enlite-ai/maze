"""Multi-step SAC implementation."""
import sys
import time
from typing import Union, Optional, BinaryIO, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.structured_env import ActorID
from maze.core.log_stats.log_stats import increment_log_step, LogStatsLevel
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.trainer import Trainer
from maze.train.utils.train_utils import compute_gradient_norm
from maze.utils.bcolors import BColors
from maze.train.parallelization.distributed_actors.base_distributed_workers_with_buffer import \
    BaseDistributedWorkersWithBuffer
from maze.train.trainers.sac.sac_algorithm_config import SACAlgorithmConfig
from maze.train.trainers.sac.sac_events import SACEvents
from torch.distributions.utils import logits_to_probs


class SAC(Trainer):
    """Multi step soft actor critic.

    :param algorithm_config: Algorithm options.
    :param learner_model: Structured torch actor critic to train.
    :param distributed_actors: Distributed actors for collection of training rollouts.
    :param model_selection: Optional model selection class, receives model evaluation results.
    :param evaluator: The evaluator to use.
    """

    def __init__(self,
                 algorithm_config: SACAlgorithmConfig,
                 learner_model: TorchActorCritic,
                 distributed_actors: BaseDistributedWorkersWithBuffer,
                 model_selection: Optional[BestModelSelection],
                 evaluator: Optional[RolloutEvaluator]):
        super().__init__(algorithm_config)

        self.algorithm_config = algorithm_config
        self.learner_model = learner_model
        self.distributed_workers = distributed_actors
        self.model_selection = model_selection
        self.evaluator = evaluator

        self.sub_step_keys = self.learner_model.policy.networks.keys()

        # initialize optimizer
        self.policy_optimizer = torch.optim.Adam(self.learner_model.policy.parameters(), lr=self.algorithm_config.lr)
        self.q_critic_optimizer = [torch.optim.Adam(critic_param, lr=self.algorithm_config.lr)
                                   for critic_param in self.learner_model.critic.per_critic_parameters()]

        # temporarily initialize env to get access to action spaces
        env = self.distributed_workers.env_factory()

        # Entropy tuning only supported for single step envs
        if self.algorithm_config.entropy_tuning:
            self.target_entropy, self.curr_log_entropy_coef = dict(), dict()
            for step_key in self.sub_step_keys:
                self.target_entropy[step_key] = dict()
                for action_key, space in env.action_spaces_dict[step_key].spaces.items():
                    if isinstance(space, spaces.Box):
                        self.target_entropy[step_key][action_key] = self.algorithm_config.target_entropy_multiplier * \
                                                                    -torch.prod(torch.Tensor(space.shape))
                    elif isinstance(space, spaces.Discrete):
                        # Entropy target is multiplied by -1 in comparison to the original paper. This make training
                        #   stable and works much much better in practice.
                        self.target_entropy[step_key][action_key] =\
                            self.algorithm_config.target_entropy_multiplier * - 0.98 * \
                            (- torch.log(torch.tensor(1.0 / space.n)))
                    else:
                        raise Exception(f'Target entropy could not be computed for the disired action space {space},'
                                        'please switch off entropy tuning and try again.')
                    print(f'Target entropy for step \'{step_key}\', action: \'{action_key}\' has been set to: '
                          f'{self.target_entropy[step_key][action_key]}')
                if not self.learner_model.critic.only_discrete_spaces[step_key]:
                    self.target_entropy[step_key] = sum(self.target_entropy[step_key].values())
                self.curr_log_entropy_coef[step_key] = torch.zeros(1, requires_grad=True,
                                                                   device=self.learner_model.device)
            self.entropy_optimizer = torch.optim.Adam(list(self.curr_log_entropy_coef.values()),
                                                      lr=self.algorithm_config.entropy_coef_lr, eps=1e-4)
            self.curr_entropy_coef = {step_key: torch.exp(log_alpha.detach()) for step_key, log_alpha in
                                      self.curr_log_entropy_coef.items()}
        else:
            if isinstance(algorithm_config.entropy_coef, list):
                self.curr_entropy_coef = {step_key: torch.tensor(entropy_coef).to(self.learner_model.device) for
                                          step_key, entropy_coef in
                                          zip(self.sub_step_keys, algorithm_config.entropy_coef)}
            elif isinstance(algorithm_config.entropy_coef, dict):
                self.curr_entropy_coef = {step_key: torch.tensor(entropy_coef).to(self.learner_model.device) for
                                          step_key, entropy_coef in algorithm_config.entropy_coef.items()}
            else:
                self.curr_entropy_coef = {
                    step_key: torch.tensor(algorithm_config.entropy_coef).to(self.learner_model.device) for
                    step_key in self.sub_step_keys}

        # Hook impala events to the event aggregator created in the super()
        self.events = self.distributed_workers.get_epoch_stats_aggregator().create_event_topic(SACEvents)

    def evaluate(self) -> None:
        """Perform evaluation on eval env.
        """
        self.evaluator.evaluate(self.learner_model.policy)

    @override(Trainer)
    def train(self, n_epochs: Optional[int] = None) -> None:
        """Train function that wraps normal train function in order to close all processes properly

        :param n_epochs: number of epochs to train.
        """

        n_epochs = self.algorithm_config.n_epochs if n_epochs is None else n_epochs

        # init minimum best model selection for early stopping
        if self.model_selection is None:
            self.model_selection = BestModelSelection(dump_file=None, model=None)

        try:
            self.distributed_workers.broadcast_updated_policy(self.learner_model.policy.state_dict())
            self.distributed_workers.start()
            self._train_async(n_epochs=n_epochs)
            self.distributed_workers.stop()
        except Exception as e:
            print('caught exception')
            self.distributed_workers.stop()
            raise e

    def load_state_dict(self, state_dict: Dict) -> None:
        """Set the model and optimizer state.

        :param state_dict: The state dict.
        """
        self.learner_model.load_state_dict(state_dict)
        self.distributed_workers.broadcast_updated_policy(self.learner_model.policy.state_dict())

    @override(Trainer)
    def state_dict(self):
        """implementation of :class:`~maze.train.trainers.common.trainer.Trainer`
        """
        return self.learner_model.state_dict()

    @override(Trainer)
    def load_state(self, file_path: Union[str, BinaryIO]) -> None:
        """implementation of :class:`~maze.train.trainers.common.trainer.Trainer`
        """
        state_dict = torch.load(file_path, map_location=torch.device(self.learner_model.device))
        self.load_state_dict(state_dict)

    def _train_async(self, n_epochs) -> None:
        """Train policy using the synchronous advantage actor critic.

        :param n_epochs: number of epochs to train.
        """

        # run training epochs
        if n_epochs <= 0:
            n_epochs = sys.maxsize
        epoch_length = self.algorithm_config.epoch_length
        patience = self.algorithm_config.patience

        # Perform a hard update on the critic
        self.learner_model.critic.update_target_weights(1.0)

        # run training epochs
        for epoch in range(n_epochs):
            start = time.time()
            print("Update epoch - {}".format(epoch))

            # compute evaluation reward
            reward = -np.inf
            if self.evaluator:
                self.evaluate()
            # take training reward and notify model selection
            else:
                if epoch > 0:
                    prev_reward = reward
                    try:
                        reward = self.distributed_workers.get_stats_value(BaseEnvEvents.reward, LogStatsLevel.EPOCH,
                                                                          name="mean")
                    except KeyError:
                        reward = prev_reward

                # best model selection
                self.model_selection.update(reward)

            # evaluate policy
            time_evaluation = time.time() - start

            # early stopping
            if patience and self.model_selection.last_improvement > patience:
                BColors.print_colored("-> no improvement since {} epochs: EARLY STOPPING!".format(patience),
                                      color=BColors.WARNING)
                increment_log_step()
                break

            time_deq_actors = 0
            time_before_update = time.time()
            for epoch_step_idx in range(epoch_length):
                q_size_before, q_size_after, time_deq_actors = self.distributed_workers.collect_rollouts()

                # Record the queue sizes
                self.events.estimated_queue_sizes(after=q_size_after, before=q_size_before)

                # policy update
                for batch_updates in range(self.algorithm_config.num_batches_per_iter):
                    self._update()
                    total_num_batch_updates =\
                        (batch_updates + epoch_step_idx * self.algorithm_config.num_batches_per_iter +
                         (epoch_length * self.algorithm_config.num_batches_per_iter) * epoch)
                    if total_num_batch_updates % self.algorithm_config.target_update_interval == 0:
                        self.learner_model.critic.update_target_weights(self.algorithm_config.tau)

                    self.distributed_workers.broadcast_updated_policy(self.learner_model.policy.state_dict())
            time_updating = time.time() - time_before_update

            total_time = time.time() - start
            self.events.time_dequeuing_actors(time=time_deq_actors, percent=time_deq_actors / total_time)

            # Buffer events
            self.events.buffer_size(len(self.distributed_workers.replay_buffer))
            self.events.buffer_avg_pick_per_transition(
                value=self.distributed_workers.replay_buffer.cum_moving_avg_num_picks)

            # increase step counter (which in turn triggers the log statistics writing)
            increment_log_step()

            print("Time required for epoch: {:.2f}s".format(total_time))
            print(' - total ({} steps) updating: {:.2f}s ({:.2f}%), mean time/step: {:.2f}s'.format(
                epoch_length * self.algorithm_config.num_batches_per_iter, time_updating, time_updating / total_time,
                time_updating / (epoch_length * self.algorithm_config.num_batches_per_iter)))
            print(' - total time evaluating the model: {:.2f}s ({:.2f}%)'.format(time_evaluation,
                                                                                 time_evaluation / total_time))

    def _update(self) -> None:
        """Perform update.

        That is collect the actor trajectories, compute the learner action logits, compute the loss
        and backprob it thought the networks.
        """

        start_update_time = time.time()

        # Collect batch from buffer ====================================================================================
        worker_output = self.distributed_workers.sample_batch(learner_device=self.learner_model.device)
        assert isinstance(worker_output, StructuredSpacesRecord)
        after_collection_time = time.time()

        # ==============================================================================================================
        # Compute critic losses ========================================================================================
        # ==============================================================================================================

        q_losses, q_values_mean = self._compute_critic_loss(worker_output)

        # NOTE: SPINNING UP IS ONLY USING ONE OPTIMIZER FOR BOTH Q NETWORKS!!!!!
        for q_optimizer, q_loss, q_params in zip(self.q_critic_optimizer, q_losses,
                                                 self.learner_model.critic.per_critic_parameters()):
            loss_per_critic = torch.stack(list(q_loss.values())).sum(0)
            q_optimizer.zero_grad()
            loss_per_critic.backward(retain_graph=False)
            if self.algorithm_config.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(q_params, self.algorithm_config.max_grad_norm)
            q_optimizer.step()

        # ==============================================================================================================
        # Compute policy losses ========================================================================================
        # ==============================================================================================================

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for q_params in self.learner_model.critic.per_critic_parameters():
            for q_param in q_params:
                q_param.requires_grad = False

        policy_losses, probs_policies, logp_policies, learner_entropies = self._compute_policy_loss(worker_output)

        policy_loss = sum(policy_losses.values())
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.algorithm_config.max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.learner_model.policy.parameters(), self.algorithm_config.max_grad_norm)
        self.policy_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next step.
        for q_params in self.learner_model.critic.per_critic_parameters():
            for q_param in q_params:
                q_param.requires_grad = True

        # ==============================================================================================================
        # Compute entropy loss =========================================================================================
        # ==============================================================================================================
        entropy_losses = 0
        if self.algorithm_config.entropy_tuning:
            entropy_losses = self._compute_entropy_loss(probs_policies, logp_policies)

            entropy_loss = sum(entropy_losses.values())
            self.entropy_optimizer.zero_grad()
            entropy_loss.backward()
            self.entropy_optimizer.step()
            self.curr_entropy_coef = {step_key: torch.exp(log_alpha.detach()) for step_key, log_alpha in
                                      self.curr_log_entropy_coef.items()}

        # ==============================================================================================================
        # Collect stats and log ========================================================================================
        # ==============================================================================================================

        # Policy data
        # for step_key in self.sub_step_keys:
        for step_key in self.sub_step_keys:
            policy_grad_norm = compute_gradient_norm(self.learner_model.policy.networks[step_key].parameters())
            # log grad norms
            self.events.policy_grad_norm(step_key=step_key, value=policy_grad_norm)

            self.events.policy_loss(step_key=step_key, value=policy_losses[step_key].detach().item())
            self.events.policy_entropy(step_key=step_key, value=learner_entropies[step_key].mean().detach().item())
            if self.learner_model.critic.only_discrete_spaces[step_key]:
                value = torch.stack(list(logp_policies[step_key].values())).sum(0).mean().detach().item()
            else:
                value = logp_policies[step_key].mean().detach().item()
            self.events.policy_mean_logp(step_key=step_key, value=value)

        # Critic data
        for critic_key in self.learner_model.critic.step_critic_keys:
            for idx, critic_step_key in enumerate(self.learner_model.critic.critic_key_mapping[critic_key]):
                critic_grad_norm = compute_gradient_norm(
                    self.learner_model.critic.networks[critic_step_key].parameters())
                self.events.critic_value(critic_key=critic_step_key, value=q_values_mean[idx][critic_key])
                self.events.critic_value_loss(critic_key=critic_step_key,
                                              value=q_losses[idx][critic_key].detach().item())
                self.events.critic_grad_norm(critic_key=critic_step_key, value=critic_grad_norm)

            # self.events.errors_between_critics(critic_key=critic_key,
            #                                    value=error_between_critics[critic_key].mean().item())

        # Entropy coef data
        for step_key in self.sub_step_keys:
            if self.algorithm_config.entropy_tuning:
                self.events.entropy_loss(step_key=step_key, value=entropy_losses[step_key].detach().cpu().item())
            self.events.entropy_coef(step_key=step_key, value=self.curr_entropy_coef[step_key].cpu().item())

        # Timing events
        total_update_time = time.time() - start_update_time
        time_backprob = time.time() - after_collection_time
        self.events.time_backprob(time=time_backprob, percent=time_backprob / total_update_time)
        time_collecting_actors_total = after_collection_time - start_update_time
        self.events.time_sampling_from_buffer(time=time_collecting_actors_total,
                                              percent=time_collecting_actors_total / total_update_time)

    def _compute_critic_loss(self, worker_output: StructuredSpacesRecord) -> \
            Tuple[List[Dict[Union[str, int], torch.Tensor]], List[Dict[Union[str, int], torch.Tensor]]]:
        """Compute the critic losses.

        :param worker_output: The batched output of the workers.
        :return: Return the critic losses as a list w.r.t. the number of critics used + mean values for stats.
        """
        next_actions = dict()
        next_actions_logits = dict()
        next_action_log_probs = dict()

        q_values_selected = self.learner_model.critic.predict_q_values(worker_output.observations_dict,
                                                                       worker_output.actions_dict, gather_output=True)
        q_values_mean = {step_key: [curr_q.detach().mean().item() if isinstance(curr_q, torch.Tensor) else
                                    torch.stack(list(curr_q.values())).mean(dim=1).detach().mean().item()
                                    for curr_q in q_values_list] for
                         step_key, q_values_list in q_values_selected.items()}

        with torch.no_grad():
            for step_key in self.sub_step_keys:
                next_policy_output = self.learner_model.policy.compute_substep_policy_output(
                    worker_output.next_observations_dict[step_key], actor_id=ActorID(step_key, 0))
                next_action = next_policy_output.prob_dist.sample()

                next_action_log_probs[step_key] = next_policy_output.prob_dist.log_prob(next_action)
                next_actions_logits[step_key] = next_policy_output.action_logits
                next_actions[step_key] = next_action

            next_q_values = self.learner_model.critic.predict_next_q_values(worker_output.next_observations_dict,
                                                                            next_actions, next_actions_logits,
                                                                            next_action_log_probs,
                                                                            self.curr_entropy_coef)

            target_q_values = dict()

            # TODO: Take into account all rewards, not just from the last sub-step
            last_rewards = list(worker_output.rewards_dict.values())[-1]
            last_dones = list(worker_output.dones_dict.values())[-1]

            for step_key, next_q_value_per_step in next_q_values.items():
                if self.learner_model.critic.only_discrete_spaces[step_key]:
                    assert isinstance(next_q_value_per_step, dict)
                    target_q_values[step_key] = {action_key: (last_rewards + (~last_dones).float() *
                                                              self.algorithm_config.gamma * next_action_q_value)
                                                 for action_key, next_action_q_value in next_q_value_per_step.items()}
                else:
                    assert isinstance(next_q_value_per_step, torch.Tensor)
                    target_q_values[step_key] = (last_rewards + (~last_dones).float() * self.algorithm_config.gamma *
                                                 next_q_value_per_step)

        q_losses = dict()
        for step_key in q_values_selected:
            per_critic_values = list()
            for q_values_per_sub_critic in q_values_selected[step_key]:
                target_q_values_per_step = target_q_values[step_key]
                if self.learner_model.critic.only_discrete_spaces[step_key]:
                    assert isinstance(q_values_per_sub_critic, dict)
                    per_action_per_critic_loss = list()
                    for action_key, q_values_per_action in q_values_per_sub_critic.items():
                        org_action_key = action_key.replace('_q_values', '')
                        per_action_loss = (q_values_per_action - target_q_values_per_step[org_action_key]).pow(2).mean()
                        per_action_per_critic_loss.append(per_action_loss)
                    # Sum the q_value of individual action in one step together
                    per_critic_values.append(torch.stack(per_action_per_critic_loss).sum(dim=0))
                else:
                    assert isinstance(q_values_per_sub_critic, torch.Tensor)
                    per_critic_values.append((q_values_per_sub_critic - target_q_values_per_step).pow(2).mean())
            q_losses[step_key] = per_critic_values

        # Transpose list of lists to get into right format and sum values from different steps together (but keep them
        #   separate w.r.t. the q network
        q_losses = [dict(zip(q_losses, t)) for t in zip(*q_losses.values())]
        q_values_mean = [dict(zip(q_values_mean, t)) for t in zip(*q_values_mean.values())]
        return q_losses, q_values_mean

    def _compute_policy_loss(self, worker_output: StructuredSpacesRecord) -> \
            Tuple[Dict[Union[str, int], torch.Tensor],
                  Dict[Union[str, int], Union[torch.Tensor, Dict[str, torch.Tensor]]],
                  Dict[Union[str, int], Union[torch.Tensor, Dict[str, torch.Tensor]]],
                  Dict[Union[str, int], torch.Tensor]]:
        """Compute the critic losses.

        :param worker_output: The batched output of the workers.
        :return: The policy losses as well a few other metrics needed for the entropy loss computation and stats.
        """

        # Sample actions and compute action log probabilities (continuous steps)/ action probabilities (discrete steps)
        policy_losses, action_entropies, action_log_probs, actions_sampled = dict(), dict(), dict(), dict()
        action_probs = dict()

        for step_key in self.sub_step_keys:
            step_obs = worker_output.observations_dict[step_key]
            learner_policy_out = self.learner_model.policy.compute_substep_policy_output(step_obs, ActorID(step_key, 0))
            learner_action = learner_policy_out.prob_dist.sample()

            # Average the logp_policy of all actions in this step (all steps if shared critic)
            if self.learner_model.critic.only_discrete_spaces[step_key]:
                probs_policy = {action_key: logits_to_probs(x) for action_key, x in
                                learner_policy_out.action_logits.items()}
                logp_policy = {action_key: torch.log(x + (x == 0.0).float() * 1e-8)
                               for action_key, x in probs_policy.items()}
            else:
                probs_policy = None
                logp_policy = torch.stack(list(learner_policy_out.prob_dist.log_prob(learner_action).values())).mean(
                    dim=0)

            action_probs[step_key] = probs_policy
            action_log_probs[step_key] = logp_policy
            actions_sampled[step_key] = learner_action
            action_entropies[step_key] = learner_policy_out.entropy

        # Predict Q values
        q_values = self.learner_model.critic.predict_q_values(worker_output.observations_dict, actions_sampled,
                                                              gather_output=False)
        if len(q_values) < len(self.sub_step_keys):
            assert len(q_values) == 1
            critic_key = list(q_values.keys())[0]
            q_values = {step_key: q_values[critic_key] for step_key in self.sub_step_keys}

        # Compute loss
        for step_key in self.sub_step_keys:
            action_log_probs_step = action_log_probs[step_key]
            q_values_step = q_values[step_key]

            if self.learner_model.critic.only_discrete_spaces[step_key]:
                action_probs_step = action_probs[step_key]

                policy_losses_per_action = list()
                # Compute the policy loss for each individual action
                for action_key in action_log_probs_step.keys():
                    q_action_key = action_key + '_q_values'
                    action_q_values = torch.stack([q_values_sub_critic[q_action_key]
                                                   for q_values_sub_critic in q_values_step]).min(dim=0).values
                    q_term = (self.curr_entropy_coef[step_key] * action_log_probs_step[action_key] - action_q_values)
                    action_policy_loss = torch.matmul(action_probs_step[action_key].unsqueeze(-2), q_term.unsqueeze(-1)
                                                      ).squeeze(-1).squeeze(-1)
                    policy_losses_per_action.append(action_policy_loss)
                # Sum the losses of all action together
                policy_losses_per_step = torch.stack(policy_losses_per_action).sum(dim=0)
                # Average the losses w.r.t. to the batch
                policy_losses[step_key] = policy_losses_per_step.mean()
            else:
                # Do not detach q_values in discrete setting
                q_value_per_step = torch.stack(q_values_step).min(dim=0).values
                # Average the losses w.r.t. to the batch
                policy_losses[step_key] = torch.mean((self.curr_entropy_coef[step_key] * action_log_probs_step -
                                                      q_value_per_step))

        return policy_losses, action_probs, action_log_probs, action_entropies

    def _compute_entropy_loss(self, action_probs: Dict[Union[str, int], Union[torch.Tensor, Dict[str, torch.Tensor]]],
                              action_log_probs: Dict[Union[str, int], Union[torch.Tensor, Dict[str, torch.Tensor]]]) \
            -> Dict[Union[str, int], torch.Tensor]:
        """Compute the entropy loss.

        :param action_probs: The probabilities of the individual actions.
        :param action_log_probs: The log probabilities of the individual actions.
        :return: The entropy loss.
        """
        entropy_losses = dict()
        for step_key in self.sub_step_keys:
            if self.learner_model.critic.only_discrete_spaces[step_key]:
                assert isinstance(self.target_entropy[step_key], dict)
                entropy_losses_per_step = list()
                for action_key, action_probs_step in action_probs[step_key].items():
                    action_log_probs_step_action = action_log_probs[step_key][action_key]
                    entropy_loss_per_action = torch.matmul(action_probs_step.unsqueeze(-2).detach(),
                                                           (-self.curr_log_entropy_coef[step_key] * (
                                                                   self.target_entropy[step_key][action_key] +
                                                                   action_log_probs_step_action).detach()).unsqueeze(
                                                               -1)
                                                           ).squeeze(-1).squeeze(-1)
                    entropy_losses_per_step.append(entropy_loss_per_action)
                # Sum together all action heads and average over batch
                entropy_losses[step_key] = torch.stack(entropy_losses_per_step).sum(0).mean()
            else:
                assert not isinstance(self.target_entropy[step_key], dict)
                entropy_losses[step_key] = torch.mean(-self.curr_log_entropy_coef[step_key] *
                                                      (self.target_entropy[step_key] + action_log_probs[
                                                          step_key]).detach())
        return entropy_losses
