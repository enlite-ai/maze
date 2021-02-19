"""Multi-step Actor Critic implementation."""
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Union, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing.io import BinaryIO

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats import increment_log_step, LogStatsLevel
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.distributions.dict import DictProbabilityDistribution
from maze.perception.perception_utils import convert_to_torch
from maze.train.parallelization.distributed_env.distributed_env import BaseDistributedEnv
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.train.trainers.common.actor_critic.actor_critic_events import MultiStepActorCriticEvents
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.trainer import Trainer
from maze.train.trainers.ppo.ppo_algorithm_config import PPOAlgorithmConfig
from maze.train.utils.train_utils import stack_numpy_dict_list, compute_gradient_norm
from maze.utils.bcolors import BColors


class MultiStepActorCritic(Trainer, ABC):
    """Base class for multi step actor critic.

    :param algorithm_config: Algorithm parameters.
    :param env: Distributed structured environment
    :param eval_env: Evaluation distributed structured environment
    :param model: Structured torch actor critic model.
    :param initial_state: path to initial state (policy weights, critic weights, optimizer state)
    :param model_selection: Optional model selection class, receives model evaluation results.
    """

    def __init__(self,
                 algorithm_config: Union[A2CAlgorithmConfig, PPOAlgorithmConfig],
                 env: Union[BaseDistributedEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv],
                 eval_env: [BaseDistributedEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv],
                 model: TorchActorCritic,
                 model_selection: Optional[BestModelSelection],
                 initial_state: Optional[str] = None):
        self.algorithm_config = algorithm_config
        self.env = env
        self.eval_env = eval_env

        self.model_selection = model_selection
        self.initial_state = initial_state

        self.step_action_keys = dict()
        for step_id, step_space in self.env.action_spaces_dict.items():
            self.step_action_keys[step_id] = list(step_space.spaces.keys())

        # initialize policies and critic
        self.model = model
        self.model.to(self.algorithm_config.device)

        # infer number of env sub-steps from the number of policies given
        self.num_env_sub_steps = len(self.model.policy.networks)
        self.sub_step_keys = list(self.model.policy.networks.keys())

        # initialize observation stack
        self.prev_obs_1 = None

        # initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.algorithm_config.lr)

        # inject statistics directly into the epoch log
        epoch_stats = env.get_stats(LogStatsLevel.EPOCH)
        self.ac_events = epoch_stats.create_event_topic(MultiStepActorCriticEvents)

    def train(self) -> None:
        """Train policy using the synchronous advantage actor critic."""

        # init minimum best model selection for early stopping
        if self.model_selection is None:
            self.model_selection = BestModelSelection(dump_file=None, model=None)

        # load initial policy weights
        if self.initial_state:
            self.load_state(self.initial_state)

        # preserve original training coef setting
        value_loss_coef = self.algorithm_config.value_loss_coef
        policy_loss_coef = self.algorithm_config.policy_loss_coef
        entropy_coef = self.algorithm_config.entropy_coef

        # run training epochs
        if self.algorithm_config.n_epochs <= 0:
            self.algorithm_config.n_epochs = sys.maxsize

        for epoch in range(self.algorithm_config.n_epochs):
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
            if self.algorithm_config.eval_repeats > 0:
                self.evaluate(repeats=self.algorithm_config.eval_repeats,
                              deterministic=self.algorithm_config.deterministic_eval)
                reward = self.eval_env.get_stats_value(BaseEnvEvents.reward, LogStatsLevel.EPOCH, name="mean")
            # take training reward
            else:
                if epoch > 0:
                    prev_reward = reward
                    try:
                        reward = self.env.get_stats_value(BaseEnvEvents.reward, LogStatsLevel.EPOCH, name="mean")
                    except:
                        reward = prev_reward

            # best model selection
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

            # increase step counter (which in turn triggers the log statistics writing)
            increment_log_step()
            epoch_time = time.time() - start
            print("Time required for epoch: {:.2f}s".format(epoch_time))
            self.ac_events.time_epoch(epoch_time)

    def evaluate(self, deterministic: bool, repeats: int) -> None:
        """Perform evaluation on eval env.

        :param deterministic: deterministic or stochastic action sampling (selection)
        :param repeats: number of evaluation episodes to average over
        """

        dones_count = 0
        prev_obs_1 = self.eval_env.reset()
        dones = np.stack([False])
        while dones_count < repeats:

            # set initial observation
            obs = prev_obs_1

            # iterate environment steps
            for step_id in self.model.policy.networks.keys():
                sampled_action = self.model.policy.compute_action(obs, policy_id=step_id, deterministic=deterministic)

                # take env step
                actions_dict_list = self._compile_actions_dict_list(sampled_action)
                obs, step_rewards, dones, infos = self.eval_env.step(actions_dict_list)

            # the last observation of the step sequence is the first observation of the next iteration
            prev_obs_1 = obs

            if np.any(dones):
                dones_count += np.count_nonzero(dones)

        # enforce the epoch stats calculation instead of waiting for the next increment_log_step() call
        self.eval_env.write_epoch_stats()

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

    def _action_log_probs_and_dists(self,
                                    obs: Dict[Union[str, int], Dict[str, Union[np.ndarray, torch.Tensor]]],
                                    actions: Dict[Union[int, str], Dict[str, Union[np.ndarray, torch.Tensor]]]) \
            -> Tuple[Dict[Union[str, int], Dict[str, torch.Tensor]],
                     Dict[Union[str, int], DictProbabilityDistribution]]:
        """Computes action log probabilities and corresponding action distributions for all sub-steps and actions.

        :param obs: Dictionary holding the sub-step observations as tensors.
        :param actions: Dictionary holding the sub-step actions as tensors.
        :return: A tuple containing the action log-probabilities and corresponding action distributions.
        """

        # convert actions to torch tensors
        actions = convert_to_torch(actions, device=self.algorithm_config.device, cast=None, in_place=True)

        # iterate sub-steps
        action_log_probs = dict()
        step_action_dists = dict()
        for step_id in self.sub_step_keys:
            # predict step action logits
            step_logits_dict = self.model.policy.compute_logits_dict(observation=obs[step_id], policy_id=step_id)

            # prepare action distributions
            prob_dist = self.model.policy.logits_dict_to_distribution(step_logits_dict, temperature=1.0)

            # compute log probs
            log_probs = prob_dist.log_prob(actions[step_id])

            # book keeping
            action_log_probs[step_id] = log_probs
            step_action_dists[step_id] = prob_dist

        return action_log_probs, step_action_dists

    def _gradient_step(self, policy_losses: List[torch.Tensor], entropies: List[torch.Tensor],
                       value_loss: torch.Tensor) -> None:
        """Perform gradient step based on given losses.

        :param policy_losses: List of policy losses.
        :param entropies: List of policy entropies.
        :param value_loss: The value loss.
        """

        # accumulate step losses
        policy_loss = sum(policy_losses)

        # compute entropy loss
        entropy_loss = sum(entropies)

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

    def _rollout(self) -> Tuple[Dict[Union[str, int], Dict[str, np.ndarray]],
                                np.array,
                                np.array,
                                Dict[Union[str, int], Dict[str, np.array]]]:
        """Perform rollout of current policy on distributed structured env env.
        """
        start_time = time.time()

        # reset environments if first rollout
        if self.prev_obs_1 is None:
            self.prev_obs_1 = self.env.reset()

        # shared lists
        rewards, dones = [], []

        # step specific lists
        step_observations = defaultdict(list)
        step_actions_taken = defaultdict(list)

        for i in range(self.algorithm_config.n_rollout_steps):
            # set initial observation
            obs = self.prev_obs_1

            # sample step actions
            assert self.num_env_sub_steps > 0
            reward, done, info = None, None, None
            for step_id in self.sub_step_keys:
                # keep observation
                step_observations[step_id].append(obs.copy())

                # sample action
                sampled_action = self.model.policy.compute_action(obs, policy_id=step_id, deterministic=False)

                # take env step
                actions_dict_list = self._compile_actions_dict_list(sampled_action)
                obs, reward, done, info = self.env.step(actions_dict_list)

                # keep action taken
                step_actions_taken[step_id].append(sampled_action)

            # update previous observation
            self.prev_obs_1 = obs

            # book keeping
            rewards.append(reward)
            dones.append(done)

        observations = dict()
        for step_id, step_obs in step_observations.items():
            observations[step_id] = stack_numpy_dict_list(step_obs, expand=True)
        rewards = np.vstack(rewards)
        dones = np.vstack(dones)
        actions_taken = dict()
        for step_id in self.sub_step_keys:
            action_dict = dict()
            for key in self.step_action_keys[step_id]:
                action_dict[key] = np.stack([step_actions_taken[step_id][s][key]
                                             for s in range(self.algorithm_config.n_rollout_steps)])
            actions_taken[step_id] = action_dict

        # log time required for rollout
        self.ac_events.time_rollout(value=time.time() - start_time)

        return observations, rewards, dones, actions_taken

    def _append_train_stats(self,
                            policy_train_stats: List[Dict[str, List[float]]],
                            critic_train_stats: List[Dict[str, List[float]]],
                            policy_losses: List[torch.Tensor],
                            entropies: List[torch.Tensor],
                            detached_values: Dict[Union[str, int], torch.Tensor],
                            value_losses: List[torch.Tensor]) -> None:
        """Append logging statistics for policies and critic.

        :param policy_train_stats: List of policy training statistics.
        :param critic_train_stats: List of critic training statistics.
        :param policy_losses: List of policy losses.
        :param entropies: List of policy entropies.
        :param detached_values: List of detached values.
        :param value_losses: List of value losses.
        """

        # policies
        for step_id in self.sub_step_keys:
            policy_train_stats[step_id]["policy_loss"].append(policy_losses[step_id].detach().item())
            policy_train_stats[step_id]["policy_entropy"].append(entropies[step_id].detach().item())

            grad_norm = compute_gradient_norm(self.model.policy.networks[step_id].parameters())
            policy_train_stats[step_id]["policy_grad_norm"].append(grad_norm)

        # critic
        for critic_id in range(self.model.critic.num_critics):
            critic_train_stats[critic_id]["critic_value"].append(detached_values[critic_id].mean().item())
            critic_train_stats[critic_id]["critic_value_loss"].append(value_losses[critic_id].detach().item())

            grad_norm = compute_gradient_norm(self.model.critic.networks[critic_id].parameters())
            critic_train_stats[critic_id]["critic_grad_norm"].append(grad_norm)

    def _log_train_stats(self, policy_train_stats: List[Dict[str, List[float]]],
                         critic_train_stats: List[Dict[str, List[float]]]) -> None:
        """Fire logging events for training statistics.

        :param policy_train_stats: List of policy training statistics.
        :param critic_train_stats: List of critic training statistics.
        """

        # log current learning rate
        self.ac_events.learning_rate(self.algorithm_config.lr)

        # policies
        for step_id in self.sub_step_keys:
            self.ac_events.policy_loss(step_id=step_id,
                                       value=np.mean(policy_train_stats[step_id]["policy_loss"]))
            self.ac_events.policy_grad_norm(step_id=step_id,
                                            value=np.mean(policy_train_stats[step_id]["policy_grad_norm"]))
            self.ac_events.policy_entropy(step_id=step_id,
                                          value=np.mean(policy_train_stats[step_id]["policy_entropy"]))

        # critic
        for critic_id in range(self.model.critic.num_critics):
            self.ac_events.critic_value(critic_id=critic_id,
                                        value=np.mean(critic_train_stats[critic_id]["critic_value"]))
            self.ac_events.critic_value_loss(critic_id=critic_id,
                                             value=np.mean(critic_train_stats[critic_id]["critic_value_loss"]))
            self.ac_events.critic_grad_norm(critic_id=critic_id,
                                            value=np.mean(critic_train_stats[critic_id]["critic_grad_norm"]))

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
