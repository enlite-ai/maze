"""Multi-step Actor Critic implementation."""
import sys
import time
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.structured_env import ActorID
from maze.core.log_stats.log_stats import increment_log_step, LogStatsLevel
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.train.parallelization.distributed_actors.distributed_actors import DistributedActors
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.train.trainers.common.actor_critic.actor_critic_events import ActorCriticEvents
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.trainer import Trainer
from maze.train.trainers.impala.impala_algorithm_config import ImpalaAlgorithmConfig
from maze.train.trainers.ppo.ppo_algorithm_config import PPOAlgorithmConfig
from maze.train.utils.train_utils import compute_gradient_norm
from maze.utils.bcolors import BColors
from tqdm import tqdm
from typing.io import BinaryIO


class ActorCritic(Trainer, ABC):
    """Base class for actor critic trainers. Suitable for multi-step and multi-agent training.

    :param algorithm_config: Algorithm parameters.
    :param rollout_generator: The rollout generator to use. This object encapsulates the env.
    :param evaluator: The evaluator to use.
    :param model: Structured torch actor critic model.
    :param model_selection: Optional model selection class, receives model evaluation results.
    """

    def __init__(
            self,
            algorithm_config: Union[A2CAlgorithmConfig, PPOAlgorithmConfig, ImpalaAlgorithmConfig],
            rollout_generator: Union[RolloutGenerator, DistributedActors],
            evaluator: Optional[RolloutEvaluator],
            model: TorchActorCritic,
            model_selection: Optional[BestModelSelection]
    ):
        super().__init__(algorithm_config)

        # initialize policies and critic
        self.model = model
        self.model.to(self.algorithm_config.device)

        # initialize rollout generator
        self.rollout_generator = rollout_generator

        # initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.algorithm_config.lr)

        # inject statistics directly into the epoch log
        epoch_stats = self.rollout_generator.get_epoch_stats_aggregator()
        self.ac_events = epoch_stats.create_event_topic(ActorCriticEvents)

        # other components
        self.model_selection = model_selection
        self.evaluator = evaluator

        if self.algorithm_config.n_epochs <= 0:
            self.algorithm_config.n_epochs = sys.maxsize

    @override(Trainer)
    def train(self, n_epochs: Optional[int] = None) -> None:
        """Main train method of the actor critic trainer. This is used in order to do algorithm specific operations
        around this method in the main train method which is called by the runner. (e.g. this is used when it comes to
        multiprocessing)

        :param n_epochs: Number of epochs to train.
        """

        n_epochs = self.algorithm_config.n_epochs if n_epochs is None else n_epochs

        # init minimum best model selection for early stopping
        if self.model_selection is None:
            self.model_selection = BestModelSelection(dump_file=None, model=None)

        # preserve original training coef setting
        value_loss_coef = self.algorithm_config.value_loss_coef
        policy_loss_coef = self.algorithm_config.policy_loss_coef
        entropy_coef = self.algorithm_config.entropy_coef

        # run training epochs
        if n_epochs <= 0:
            n_epochs = sys.maxsize

        for epoch in range(n_epochs):
            start = time.time()
            print("Update epoch - {}".format(epoch))

            # check for critic burn in and reset coefficient to only update the critic
            if epoch < self.algorithm_config.critic_burn_in_epochs:
                self.algorithm_config.value_loss_coef = 1.0
                self.algorithm_config.policy_loss_coef = 0.0
                self.algorithm_config.entropy_coef = 0.0
            else:
                self.algorithm_config.value_loss_coef = value_loss_coef
                self.algorithm_config.policy_loss_coef = policy_loss_coef
                self.algorithm_config.entropy_coef = entropy_coef

            # compute evaluation reward
            reward = -np.inf
            if self.evaluator:
                self.evaluate()
            # take training reward and notify best model selection manually
            else:
                if epoch > 0:
                    prev_reward = reward
                    try:
                        reward = self.rollout_generator.get_stats_value(BaseEnvEvents.reward, LogStatsLevel.EPOCH,
                                                                        name="mean")
                    except:
                        reward = prev_reward

                self.model_selection.update(reward)

            # early stopping
            if self.algorithm_config.patience and \
                    self.model_selection.last_improvement > self.algorithm_config.patience:
                BColors.print_colored("-> no improvement since {} epochs: EARLY STOPPING!"
                                      .format(self.algorithm_config.patience), color=BColors.WARNING)
                increment_log_step()
                break

            # policy update
            for _ in tqdm(range(self.algorithm_config.epoch_length)):
                update_start = time.time()
                self._update()
                self.ac_events.time_update(time.time() - update_start)

            epoch_time = time.time() - start
            self.ac_events.time_epoch(epoch_time)

            # increase step counter (which in turn triggers the log statistics writing)
            increment_log_step()

            print("Time required for epoch: {:.2f}s".format(epoch_time))

    def evaluate(self) -> None:
        """Perform evaluation on eval env."""
        self.evaluator.evaluate(self.model.policy)

    @override(Trainer)
    def state_dict(self):
        """implementation of :class:`~maze.train.trainers.common.trainer.Trainer`
        """
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict) -> None:
        """Set the model and optimizer state.
        :param state_dict: The state dict.
        """
        self.model.load_state_dict(state_dict)

    @override(Trainer)
    def load_state(self, file_path: Union[str, BinaryIO]) -> None:
        """implementation of :class:`~maze.train.trainers.common.trainer.Trainer`
        """
        state_dict = torch.load(file_path, map_location=torch.device(self.algorithm_config.device))
        self.load_state_dict(state_dict)

    @abstractmethod
    def _update(self) -> None:
        """Perform policy and critic update.
        """
        raise NotImplementedError

    def _gradient_step(self, policy_losses: List[torch.Tensor], entropies: List[torch.Tensor],
                       value_losses: List[torch.Tensor]) -> None:
        """Perform gradient step based on given losses.

        :param policy_losses: List of policy losses.
        :param entropies: List of policy entropies.
        :param value_losses: The value loss.
        """

        # accumulate step losses
        policy_loss = sum(policy_losses)

        # compute entropy loss
        entropy_loss = sum(entropies)

        # compute value loss
        value_loss = sum(value_losses)

        # perform update
        loss = self.algorithm_config.value_loss_coef * value_loss + self.algorithm_config.policy_loss_coef * policy_loss
        if self.algorithm_config.entropy_coef > 0.0:
            loss -= self.algorithm_config.entropy_coef * entropy_loss

        # compute backward pass
        self.optimizer.zero_grad()
        loss.backward(retain_graph=False)

        # limit to maximum allowed gradient norm
        if self.algorithm_config.max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.algorithm_config.max_grad_norm)

        # perform optimizer step
        self.optimizer.step()

    def _rollout(self) -> StructuredSpacesRecord:
        """Perform rollout of current policy on distributed structured env and log the time it took."""
        start_time = time.time()
        trajectory = self.rollout_generator.rollout(self.model.policy, n_steps=self.algorithm_config.n_rollout_steps)
        self.ac_events.time_rollout(value=time.time() - start_time)
        return trajectory.stack().to_torch(device=self.algorithm_config.device)

    def _append_train_stats(self,
                            policy_train_stats: List[Dict[str, List[float]]],
                            critic_train_stats: List[Dict[str, List[float]]],
                            actor_ids: List[ActorID],
                            policy_losses: List[torch.Tensor],
                            entropies: List[torch.Tensor],
                            detached_values: List[torch.Tensor],
                            value_losses: List[torch.Tensor]) -> None:
        """Append logging statistics for policies and critic.

        :param policy_train_stats: List of policy training statistics.
        :param critic_train_stats: List of critic training statistics.
        :param policy_losses: List of policy losses.
        :param entropies: List of policy entropies.
        :param detached_values: List of detached values.
        :param value_losses: List of value losses.
        """

        # Policies
        for actor_id, substep_loss, substep_entropies in zip(actor_ids, policy_losses, entropies):
            policy_train_stats[actor_id[0]]["policy_loss"].append(substep_loss.detach().item())
            policy_train_stats[actor_id[0]]["policy_entropy"].append(substep_entropies.detach().item())

            grad_norm = compute_gradient_norm(self.model.policy.network_for(actor_id).parameters())
            policy_train_stats[actor_id[0]]["policy_grad_norm"].append(grad_norm)

        # Critic(s)
        #  - if there is just one critic, report only values from the first sub-step.
        #  - otherwise, use sub-step keys to identify the critics.
        first_critic_id = list(self.model.critic.networks.keys())[-1]
        critic_ids = [first_critic_id] if self.model.critic.num_critics == 1 else list(map(lambda x: x[0], actor_ids))
        for critic_id, substep_detached_values, substep_losses in zip(critic_ids, detached_values, value_losses):
            critic_train_stats[critic_id]["critic_value"].append(substep_detached_values.mean().item())
            critic_train_stats[critic_id]["critic_value_loss"].append(substep_losses.detach().item())

            grad_norm = compute_gradient_norm(self.model.critic.networks[critic_id].parameters())
            critic_train_stats[critic_id]["critic_grad_norm"].append(grad_norm)

    def _log_train_stats(self, policy_train_stats: Dict[Union[str, int], Dict[str, List[float]]],
                         critic_train_stats: Dict[Union[str, int], Dict[str, List[float]]]) -> None:
        """Fire logging events for training statistics.

        :param policy_train_stats: Dict of policy training statistics.
        :param critic_train_stats: Dict of critic training statistics.
        """

        # log current learning rate
        self.ac_events.learning_rate(self.algorithm_config.lr)

        # policies
        for substep_key, stats in policy_train_stats.items():
            self.ac_events.policy_loss(substep_key=substep_key, value=np.mean(stats["policy_loss"]))
            self.ac_events.policy_grad_norm(substep_key=substep_key, value=np.mean(stats["policy_grad_norm"]))
            self.ac_events.policy_entropy(substep_key=substep_key, value=np.mean(stats["policy_entropy"]))

        # critic
        for critic_id, stats in critic_train_stats.items():
            self.ac_events.critic_value(critic_id=critic_id, value=np.mean(stats["critic_value"]))
            self.ac_events.critic_value_loss(critic_id=critic_id, value=np.mean(stats["critic_value_loss"]))
            self.ac_events.critic_grad_norm(critic_id=critic_id, value=np.mean(stats["critic_grad_norm"]))

    @classmethod
    def _normalize_advantages(cls, advantages: List[torch.Tensor]) -> List[torch.Tensor]:
        """Normalize advantages.

        :param advantages: List of advantages.
        :return: List of normalized advantages.
        """
        return [(a - a.mean()) / (a.std() + 1e-8) for a in advantages]

    @classmethod
    def _compile_actions_dict_list(cls, sampled_action: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
        action_keys = list(sampled_action.keys())
        actions_dict_list = []
        for j in range(len(sampled_action[action_keys[0]])):
            action_dict = dict()
            for k in action_keys:
                action_dict[k] = sampled_action[k][j]
            actions_dict_list.append(action_dict)
        return actions_dict_list
