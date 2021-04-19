"""Multi-step IMPALA implementation."""
import sys
import time
from typing import Union, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from typing.io import BinaryIO

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_state_critic import TorchSharedStateCritic
from maze.core.annotations import override
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats import increment_log_step, LogStatsLevel
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.train.parallelization.distributed_actors.distributed_actors import DistributedActors
from maze.train.parallelization.vector_env.vector_env import VectorEnv
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.trainer import Trainer
from maze.train.trainers.impala import impala_vtrace
from maze.train.trainers.impala.impala_algorithm_config import ImpalaAlgorithmConfig
from maze.train.trainers.impala.impala_events import MultiStepIMPALAEvents
from maze.train.trainers.impala.impala_learner import ImpalaLearner, LearnerOutput
from maze.train.utils.train_utils import compute_gradient_norm
from maze.utils.bcolors import BColors


class MultiStepIMPALA(Trainer):
    """Multi step advantage actor critic.

    :param model: Structured policy to train
    :param rollout_actors: Distributed actors for collection of training rollouts
    :param eval_env: Env to run evaluation on
    :param options: Algorithm options
    """

    def __init__(self, model: TorchActorCritic, rollout_actors: DistributedActors,
                 eval_env: Union[VectorEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv],
                 options: ImpalaAlgorithmConfig):
        super().__init__(options)

        self.distributed_actors = rollout_actors
        self.model = model

        self.learner = ImpalaLearner(eval_env, self.model, options.n_rollout_steps)

        self.lr = options.lr
        self.gamma = options.gamma
        self.policy_loss_coef = options.policy_loss_coef
        self.value_loss_coef = options.value_loss_coef
        self.entropy_coef = options.entropy_coef
        self.max_grad_norm = options.max_grad_norm
        reward_clipping_options = ['abs_one', 'soft_asymmetric', 'None']
        assert options.reward_clipping in reward_clipping_options, 'Please specify on of the possible options ' \
                                                                   'for reward clipping: {}'.format(
            reward_clipping_options)
        self.reward_clipping = options.reward_clipping

        self.vtrace_clip_rho_threshold = options.vtrace_clip_rho_threshold
        self.vtrace_clip_pg_rho_threshold = options.vtrace_clip_pg_rho_threshold

        # initialize optimizer
        self.optimizer = torch.optim.Adam(self.learner.model.parameters(), lr=self.lr)

        # Hook impala events to the event aggregator created in the super()
        self.impala_events = \
            self.distributed_actors.get_epoch_stats_aggregator().create_event_topic(MultiStepIMPALAEvents)

    def evaluate(self, deterministic: bool, repeats: int) -> None:
        """Perform evaluation on eval env.

        :param deterministic: deterministic or stochastic action sampling (selection)
        :param repeats: number of evaluation episodes to average over
        """
        self.learner.evaluate(deterministic, repeats)

    @override(Trainer)
    def train(
        self,
        n_epochs: Optional[int] = None,
        epoch_length: Optional[int] = None,
        deterministic_eval: Optional[bool] = None,
        eval_repeats: Optional[int] = None,
        patience: Optional[int] = None,
        model_selection: Optional[BestModelSelection] = None
    ) -> None:
        """Train function that wraps normal train function in order to close all processes properly

        :param n_epochs: number of epochs to train.
        :param epoch_length: number of updates per epoch.
        :param deterministic_eval: run evaluation in deterministic mode (argmax-policy)
        :param eval_repeats: number of evaluation trials
        :param patience: number of steps used for early stopping
        :param model_selection: Optional model selection class, receives model evaluation results
        """

        if epoch_length is None:
            epoch_length = self.algorithm_config.epoch_length
        if deterministic_eval is None:
            deterministic_eval = self.algorithm_config.deterministic_eval
        if eval_repeats is None:
            eval_repeats = self.algorithm_config.eval_repeats

        try:
            self.distributed_actors.broadcast_updated_policy(self.model.state_dict())
            self.distributed_actors.start()
            self.train_async(n_epochs, epoch_length, deterministic_eval, eval_repeats, patience, model_selection)
            self.distributed_actors.stop()
        except Exception as e:
            print('caught exception')
            self.distributed_actors.stop()
            raise e

    def train_async(
        self,
        n_epochs: Optional[int] = None,
        epoch_length: Optional[int] = None,
        deterministic_eval: Optional[bool] = None,
        eval_repeats: Optional[int] = None,
        patience: Optional[int] = None,
        model_selection: Optional[BestModelSelection] = None
    ) -> None:
        """Train policy using the synchronous advantage actor critic.

        :param n_epochs: number of epochs to train.
        :param epoch_length: number of updates per epoch.
        :param deterministic_eval: run evaluation in deterministic mode (argmax-policy)
        :param eval_repeats: number of evaluation trials
        :param patience: number of steps used for early stopping
        :param model_selection: Optional model selection class, receives model evaluation results
        """

        if epoch_length is None:
            epoch_length = self.algorithm_config.epoch_length
        if deterministic_eval is None:
            deterministic_eval = self.algorithm_config.deterministic_eval
        if eval_repeats is None:
            eval_repeats = self.algorithm_config.eval_repeats
        # init minimum best model selection for early stopping
        if model_selection is None:
            model_selection = BestModelSelection(dump_file=None, model=None)

        # run training epochs
        if n_epochs <= 0:
            n_epochs = sys.maxsize

        # run training epochs
        for epoch in range(n_epochs):
            start = time.time()
            print("Update epoch - {}".format(epoch))

            if eval_repeats > 0:
                self.evaluate(repeats=eval_repeats, deterministic=deterministic_eval)
                reward = self.learner.env.get_stats_value(BaseEnvEvents.reward, LogStatsLevel.EPOCH, name="mean")
            # take training reward
            else:
                reward = -np.inf if epoch < 1 else \
                    self.distributed_actors.get_stats_value(BaseEnvEvents.reward, LogStatsLevel.EPOCH,
                                                            name="mean")

            # evaluate policy
            time_evaluation = time.time() - start

            # best model selection
            model_selection.update(reward)

            # early stopping
            if patience and model_selection.last_improvement > patience:
                BColors.print_colored("-> no improvement since {} epochs: EARLY STOPPING!".format(patience),
                                      color=BColors.WARNING)
                increment_log_step()
                break

            # policy update
            time_before_update = time.time()
            for epoch_step_idx in range(epoch_length):
                self._update(epoch, epoch_step_idx)
                self.distributed_actors.broadcast_updated_policy(self.model.state_dict())
            time_updating = time.time() - time_before_update

            # increase step counter (which in turn triggers the log statistics writing)
            increment_log_step()
            total_time = time.time() - start
            print("Time required for epoch: {:.2f}s".format(total_time))
            print(' - total ({} steps) updating: {:.2f}s ({:.2f}%), mean time/step: {:.2f}s'.format(
                epoch_length, time_updating, time_updating / total_time, time_updating / epoch_length))
            print(' - total time evaluating the model: {:.2f}s ({:.2f}%)'.format(time_evaluation,
                                                                                 time_evaluation / total_time))

    def load_state_dict(self, state_dict: Dict) -> None:
        """Set the model and optimizer state.
        :param state_dict: The state dict.
        """
        self.model.load_state_dict(state_dict)

    @override(Trainer)
    def load_state(self, file_path: Union[str, BinaryIO]) -> None:
        """implementation of :class:`~maze.train.trainers.common.trainer.Trainer`
        """
        state_dict = torch.load(file_path, map_location=torch.device(self.model.device))
        self.load_state_dict(state_dict)

    @classmethod
    def _shift_outputs(cls, step_record: StructuredSpacesRecord, learner_output: LearnerOutput) -> Tuple[StructuredSpacesRecord,
                                                                                                         LearnerOutput]:
        """Shift the output of the actors and agents by removing the last element

        :param step_record: the output of the actors already batched
        :param learner_output: the output of the leaner

        :return: The shifted actor and learner output
        """
        for substep_record in step_record.substep_records:
            substep_record.reward = substep_record.reward[:-1]
            substep_record.done = substep_record.done[:-1]

            for name in ['observation', 'action', 'logits']:
                field = getattr(substep_record, name)
                for key, value in field.items():
                    getattr(substep_record, name)[key] = value[:-1]

        new_learner_output = {}
        for name in ['values', 'detached_values', 'actions_logits']:
            field = getattr(learner_output, name)
            new_learner_output[name] = field
            for step_idx, value in field.items():
                if name in ['values', 'detached_values']:
                    new_learner_output[name][step_idx] = value[:-1]
                else:
                    for key, val in field[step_idx].items():
                        new_learner_output[name][step_idx][key] = val[:-1]

        return step_record, LearnerOutput(**new_learner_output)

    def _update(self, epoch_idx: int, epoch_step_idx: int) -> None:
        """Perform update. That is collect the actor trajectories, compute the learner action logits, compute the loss
            and backprob it thought the networks

        :param epoch_idx: The current epoch index
        :param epoch_step_idx: The current step w.r.t. to the current epoch
        """

        # Collect self.actor_batch_size actor outputs from the queue (all in time major - 1 dim rollout_length,
        #                                                                                 2 dim actor_batch_size..
        start_update_time = time.time()

        record, q_size_before, q_size_after, time_deq_actors = self.distributed_actors.collect_outputs(
            self.learner.model.device)
        after_collection_time = time.time()

        # Record the queue sizes
        self.impala_events.estimated_queue_sizes(after=q_size_after, before=q_size_before)

        # Compute learner rollout and return values (for one rollout - no batch dimensions in the tensors)
        learner_output: LearnerOutput = self.learner.learner_rollout_on_agent_output(record)
        after_learner_rollout_time = time.time()

        # Test if Agents output is already in time major
        for output_field in [record.logits_dict, record.actions_dict, record.observations_dict,
                             learner_output.actions_logits]:
            assert isinstance(output_field, dict) \
                   and set(output_field) == set(self.learner.sub_step_keys), "acton_output filed should be a list of " \
                                                                             "len env.num_substeps"
            for _, step_dict in output_field.items():
                for key, value in step_dict.items():
                    assert isinstance(value, torch.Tensor), 'actor output field should all be torch.Tesors, ' \
                                                            'not so: {}, {}'.format(key, value)
                    assert value.shape[0] == self.learner.n_rollout_steps, 'First dimension should be the time ' \
                                                                           'dimension '
                    assert value.shape[1] == self.distributed_actors.batch_size, \
                        'Second dimension should be the batch dimension'

        bootstrap_value = {step_key: step_values[-1]
                           for step_key, step_values in learner_output.detached_values.items()}

        # Shift values:
        record, learner_output = self._shift_outputs(record, learner_output)

        # TODO: Take into account all rewards, not just from the last sub-step
        last_rewards = list(record.rewards_dict.values())[-1]

        # Clip reward:
        if self.reward_clipping == 'abs_one':
            clipped_rewards = last_rewards.clamp(-1, 1)
        elif self.reward_clipping == 'soft_asymmetric':
            squeezed = torch.tanh(last_rewards / 5.0)
            clipped_rewards = torch.where(last_rewards < 0, 0.3 * squeezed, squeezed) * 5.
        else:
            clipped_rewards = record.rewards_dict[list(record.rewards_dict.keys())[-1]]
        # START: Loss computation --------------------------------------------------------------------------------------
        vtrace_returns = impala_vtrace.from_logits(
            behaviour_policy_logits=record.logits_dict,
            target_policy_logits=learner_output.actions_logits,
            actions=record.actions_dict,
            action_spaces=self.learner.env.action_spaces_dict,
            distribution_mapper=self.model.policy.distribution_mapper,
            discounts=(~record.dones_dict[list(record.dones_dict.keys())[-1]]).float() * self.gamma,
            rewards=clipped_rewards,
            values=learner_output.detached_values,
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=self.vtrace_clip_rho_threshold,
            clip_pg_rho_threshold=self.vtrace_clip_pg_rho_threshold,
            device=self.learner.model.device)

        # Compute loss as a weighted sum of the baseline loss, the policy gradient
        # loss and an entropy regularization term.

        # The policy gradients loss
        policy_losses = dict()
        for step_key in self.learner.sub_step_keys:
            step_policy_loss = torch.tensor(0.0).to(self.learner.model.device)
            for action_key in self.learner.step_action_keys[step_key]:
                step_policy_loss -= (vtrace_returns.pg_advantages[step_key] *
                                     vtrace_returns.target_action_log_probs[step_key][action_key]).mean()
            policy_losses[step_key] = step_policy_loss
        overall_policy_loss = sum(policy_losses.values())

        # compute value loss
        value_losses = dict()
        for step_key in self.learner.sub_step_keys[:len(self.learner.model.critic.networks)]:
            value_loss = (vtrace_returns.vs[step_key] -
                          learner_output.values[step_key]).pow(2).mean()
            value_losses[step_key] = value_loss
        overall_value_loss = sum(value_losses.values())

        # compute entropy loss
        entropy_losses = {step_key: action_dist.entropy().mean()
                          for step_key, action_dist in vtrace_returns.target_step_action_dists.items()}
        overall_entropy_loss = sum(entropy_losses.values())

        # The summed weighted loss
        total_loss = (self.policy_loss_coef * overall_policy_loss + self.value_loss_coef * overall_value_loss -
                      self.entropy_coef * overall_entropy_loss)

        after_loss_computation_time = time.time()
        # END: Loss computation ----------------------------------------------------------------------------------------

        # compute backward pass
        total_loss.backward()

        # # limit to maximum allowed gradient norm
        if self.max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.learner.model.parameters(), self.max_grad_norm)

        # perform optimizer step
        self.optimizer.step()

        # estimate gradient norms
        policy_grad_norms, critic_grad_norms = dict(), dict()
        for step_key in self.learner.sub_step_keys:
            policy_grad_norms[step_key] = compute_gradient_norm(
                self.learner.model.policy.networks[step_key].parameters())
        for critic_key in list(self.learner.model.critic.networks.keys()):
            critic_grad_norms[critic_key] = compute_gradient_norm(
                self.learner.model.critic.networks[critic_key].parameters())

        for step_key in self.learner.sub_step_keys:
            self.impala_events.policy_loss(step_key=step_key, value=policy_losses[step_key].detach().item())
            self.impala_events.policy_grad_norm(step_key=step_key, value=policy_grad_norms[step_key])
            self.impala_events.policy_entropy(step_key=step_key, value=entropy_losses[step_key].detach().item())

        for step_key in self.learner.sub_step_keys[:len(self.learner.model.critic.networks)]:
            critic_key = 0 if isinstance(self.learner.model.critic, TorchSharedStateCritic) else step_key
            self.impala_events.critic_value(critic_key=critic_key,
                                            value=vtrace_returns.vs[step_key].mean().item())
            self.impala_events.critic_value_loss(critic_key=critic_key,
                                                 value=value_losses[step_key].detach().item())
            self.impala_events.critic_grad_norm(critic_key=critic_key, value=critic_grad_norms[critic_key])

        total_update_time = time.time() - start_update_time
        time_backprob = time.time() - after_loss_computation_time
        self.impala_events.time_backprob(time=time_backprob, percent=time_backprob / total_update_time)
        time_collecting_actors_total = after_collection_time - start_update_time
        self.impala_events.time_collecting_actors(time=time_collecting_actors_total,
                                                  percent=time_collecting_actors_total / total_update_time)
        self.impala_events.time_dequeuing_actors(time=time_deq_actors,
                                                 percent=time_deq_actors / total_update_time)
        time_learner_rollout = after_learner_rollout_time - after_collection_time
        self.impala_events.time_learner_rollout(time=time_learner_rollout,
                                                percent=time_learner_rollout / total_update_time)
        time_loss_computation = after_loss_computation_time - after_learner_rollout_time
        self.impala_events.time_loss_computation(time=time_loss_computation,
                                                 percent=time_loss_computation / total_update_time)
