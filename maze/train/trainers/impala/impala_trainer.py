"""Multi-step IMPALA implementation."""
import time
from collections import defaultdict
from typing import Optional

import torch
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.perception.perception_utils import map_nested_structure
from maze.train.parallelization.distributed_actors.distributed_actors import DistributedActors
from maze.train.trainers.common.actor_critic.actor_critic_trainer import ActorCritic
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.trainer import Trainer
from maze.train.trainers.impala import impala_vtrace
from maze.train.trainers.impala.impala_algorithm_config import ImpalaAlgorithmConfig
from maze.train.trainers.impala.impala_events import ImpalaEvents


class IMPALA(ActorCritic):
    """Multi step advantage actor critic.
    """

    def __init__(self,
                 algorithm_config: ImpalaAlgorithmConfig,
                 rollout_generator: DistributedActors,
                 evaluator: Optional[RolloutEvaluator],
                 model: TorchActorCritic,
                 model_selection: Optional[BestModelSelection]):
        super().__init__(algorithm_config, rollout_generator, evaluator, model, model_selection)

        # inject statistics directly into the epoch log
        epoch_stats = self.rollout_generator.get_epoch_stats_aggregator()
        self.impala_events = epoch_stats.create_event_topic(ImpalaEvents)

    @override(Trainer)
    def train(self, n_epochs: Optional[int] = None) -> None:
        """Train function that wraps normal train function in order to close all processes properly

        :param n_epochs: number of epochs to train.
        """

        try:
            self.rollout_generator.broadcast_updated_policy(self.model.state_dict())
            self.rollout_generator.start()
            super().train(n_epochs)
            self.rollout_generator.stop()
        except Exception as e:
            print('caught exception')
            self.rollout_generator.stop()
            raise e

    @override(ActorCritic)
    def _rollout(self) -> StructuredSpacesRecord:
        """Perform rollout of current policy on distributed structured env and log the time it took."""
        raise NotImplementedError

    @override(ActorCritic)
    def _update(self) -> None:
        """Perform update. That is collect the actor trajectories, compute the learner action logits, compute the loss
            and backprob it thought the networks
        """

        # Collect self.actor_batch_size actor outputs from the queue (all in time major - 1 dim rollout_length,
        #                                                                                 2 dim actor_batch_size..
        start_update_time = time.time()

        record, q_size_before, q_size_after, time_deq_actors = self.rollout_generator.collect_outputs(self.model.device)
        after_collection_time = time.time()

        # Record the queue sizes
        self.impala_events.estimated_queue_sizes(after=q_size_after, before=q_size_before)

        # Compute learner rollout and return values (for one rollout - no batch dimensions in the tensors)
        learner_policy_output, learner_critic_output = self.model.compute_actor_critic_output(record)
        after_learner_rollout_time = time.time()

        # Take bootstrap return value (last value of each substep)
        bootstrap_value = [step_values[-1] for step_values in learner_critic_output.values]

        # TODO: Take into account all rewards, not just from the last sub-step
        last_rewards = record.rewards[-1]
        discounts = (~record.dones[-1]).float() * self.algorithm_config.gamma
        # START: Loss computation --------------------------------------------------------------------------------------

        vtrace_returns = impala_vtrace.from_logits(
            behaviour_policy_logits=map_nested_structure(record.logits, lambda x: x[:-1], in_place=True),
            target_policy_logits=map_nested_structure(learner_policy_output.action_logits, lambda x: x[:-1],
                                                      in_place=True),
            actions=map_nested_structure(record.actions, lambda x: x[:-1], in_place=True),
            distribution_mapper=self.model.policy.distribution_mapper,
            discounts=map_nested_structure(discounts, lambda x: x[:-1], in_place=False),
            rewards=map_nested_structure(last_rewards, lambda x: x[:-1], in_place=False),
            values=map_nested_structure(learner_critic_output.detached_values, lambda x: x[:-1], in_place=True),
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=self.algorithm_config.vtrace_clip_rho_threshold,
            clip_pg_rho_threshold=self.algorithm_config.vtrace_clip_pg_rho_threshold,
            device=self.model.device)

        # Compute loss as a weighted sum of the baseline loss, the policy gradient
        # loss and an entropy regularization term.

        # The policy gradients loss
        policy_losses = list()
        for step_pg_adv, step_target_log_probs in zip(vtrace_returns.pg_advantages,
                                                      vtrace_returns.target_action_log_probs):
            step_p_loss = -torch.sum(
                torch.stack([(target_action_log_prob * step_pg_adv).mean() for target_action_log_prob in
                             step_target_log_probs.values()]))
            policy_losses.append(step_p_loss)

        # compute value loss
        shifted_values = map_nested_structure(learner_critic_output.values, lambda x: x[:-1], in_place=True)
        if self.model.critic.num_critics == 1:
            value_losses = [(shifted_values[0] - vtrace_returns.vs[0]).pow(2.0).mean()]
        else:
            value_losses = [(vv - vt).pow(2).mean() for vv, vt in zip(shifted_values, vtrace_returns.vs)]

        value_losses = list(map(lambda x: x / 2.0, value_losses))

        # compute entropy loss
        entropy_losses = [entropy.mean() for entropy in learner_policy_output.entropies]

        after_loss_computation_time = time.time()

        # perform gradient step
        self._gradient_step(policy_losses=policy_losses, entropies=entropy_losses, value_losses=value_losses)

        # collect training stats for logging
        policy_train_stats = defaultdict(lambda: defaultdict(list))
        critic_train_stats = defaultdict(lambda: defaultdict(list))
        self._append_train_stats(policy_train_stats, critic_train_stats,
                                 record.actor_ids, policy_losses, entropy_losses,
                                 learner_critic_output.detached_values, value_losses)

        # fire logging events
        self._log_train_stats(policy_train_stats, critic_train_stats)

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

        self.rollout_generator.broadcast_updated_policy(self.model.state_dict())
