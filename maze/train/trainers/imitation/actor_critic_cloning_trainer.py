"""Trainer class for offline actor-critic training."""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Union, Optional

import torch
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.log_stats.log_stats import LogStatsAggregator, LogStatsLevel, get_stats_logger
from maze.perception.perception_utils import convert_to_torch
from maze.train.trainers.common.evaluators.evaluator import Evaluator
from maze.train.trainers.common.trainer import Trainer
from maze.train.trainers.imitation.bc_loss import BCLoss
from maze.train.trainers.imitation.critic_cloning_loss import StateCriticCloningLoss
from maze.train.trainers.imitation.imitation_events import CriticImitationEvents, ImitationEvents
from maze.train.utils.train_utils import compute_gradient_norm
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing.io import BinaryIO


@dataclass
class ActorCriticCloningTrainer(Trainer):
    """Trainer for offline actor-critic learning (cloning).

    Runs training on top of provided trajectory data.

    In structured (multi-step) envs, all policies and critics are trained simultaneously based on the substep rewards
    and observation present in the trajectory data.
    """

    train_data_loader: DataLoader
    """Data loader for loading trajectory data."""

    actor_critic_model: TorchActorCritic
    """Structured policy to train."""

    actor_optimizer: Optimizer
    """Optimizer to use for actor"""

    critic_optimizer: Optimizer
    """Optimizer to use for critic"""

    actor_loss: BCLoss
    """Class providing the training loss function."""

    critic_loss: StateCriticCloningLoss
    """Class providing the training loss function."""

    train_stats: LogStatsAggregator = LogStatsAggregator(LogStatsLevel.EPOCH, get_stats_logger("train"))
    """Training statistics"""

    actor_events: ImitationEvents = train_stats.create_event_topic(ImitationEvents)
    """Imitation-specific training events"""

    critic_events: CriticImitationEvents = train_stats.create_event_topic(CriticImitationEvents)
    """Imitation-specific training events"""

    def train(self, n_epochs: int, evaluator: Evaluator) -> None:
        """Run training.

        :param n_epochs: How many epochs to train for
        :param evaluator: Evaluator to use for evaluation rollouts
        """

        # train for several epochs
        for _ in tqdm(range(n_epochs)):
            for iteration, data in enumerate(self.train_data_loader, 0):
                self._run_iteration(data)

        # evaluate policy
        evaluator.evaluate(self.actor_critic_model.policy)

    def load_state_dict(self, state_dict: Dict) -> None:
        """Set the model and optimizer state.

        :param state_dict: The state dict.
        """
        self.actor_critic_model.load_state_dict(state_dict)

    @override(Trainer)
    def load_state(self, file_path: Union[str, BinaryIO]) -> None:
        """implementation of :class:`~maze.train.trainers.common.trainer.Trainer`
        """
        state_dict = torch.load(file_path, map_location=torch.device(self.actor_critic_model.device))
        self.load_state_dict(state_dict)

    def _run_iteration(self, data: Tuple[Dict[Union[int, str], Any], Dict[Union[int, str], Any],
                                         Dict[Union[int, str], Any]]) -> None:
        self.actor_critic_model.policy.train()
        self.actor_critic_model.critic.train()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        observation_dict, action_dict, step_returns = data
        convert_to_torch(step_returns, device=self.actor_critic_model.device, cast=None, in_place=True)
        convert_to_torch(step_returns, device=self.actor_critic_model.device, cast=None, in_place=True)

        # update actor
        total_loss = self.actor_loss.calculate_loss(
            policy=self.actor_critic_model.policy, observation_dict=observation_dict,
            action_dict=action_dict, events=self.actor_events)
        total_loss.backward()
        self.actor_optimizer.step()

        # update critic
        total_loss = self.critic_loss.calculate_loss(
            critic=self.actor_critic_model.critic, observation_dict=observation_dict,
            step_returns=step_returns, events=self.critic_events)
        total_loss.backward()
        self.critic_optimizer.step()

        # Report additional stats
        for policy_id in observation_dict.keys():
            l2_norm = sum([param.norm() for param in self.actor_critic_model.policy.networks[policy_id].parameters()])
            grad_norm = compute_gradient_norm(self.actor_critic_model.policy.networks[policy_id].parameters())
            self.actor_events.policy_l2_norm(step_id=policy_id, value=l2_norm.item())
            self.actor_events.policy_grad_norm(step_id=policy_id, value=grad_norm)

            l2_norm = sum([param.norm() for param in self.actor_critic_model.critic.networks[policy_id].parameters()])
            grad_norm = compute_gradient_norm(self.actor_critic_model.critic.networks[policy_id].parameters())
            self.critic_events.critic_l2_norm(step_id=policy_id, value=l2_norm.item())
            self.critic_events.critic_grad_norm(step_id=policy_id, value=grad_norm)

        self.actor_critic_model.policy.eval()
        self.actor_critic_model.critic.eval()
