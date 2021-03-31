"""Trainer class for offline critic training."""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Union

import torch
from maze.core.agent.torch_state_critic import TorchStateCritic
from maze.core.annotations import override
from maze.core.log_stats.log_stats import LogStatsAggregator, LogStatsLevel, get_stats_logger, increment_log_step
from maze.perception.perception_utils import convert_to_torch
from maze.train.trainers.common.trainer import Trainer
from maze.train.trainers.imitation.critic_cloning_loss import StateCriticCloningLoss
from maze.train.trainers.imitation.imitation_events import CriticImitationEvents
from maze.train.utils.train_utils import compute_gradient_norm
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from typing.io import BinaryIO


@dataclass
class CriticCloningTrainer(Trainer):
    """Trainer for offline critic learning.

    Runs training on top of provided trajectory data.

    In structured (multi-step) envs, all critics are trained simultaneously based on the substep rewards
    and observation present in the trajectory data.
    """

    data_loader: DataLoader
    """Data loader for loading trajectory data."""

    critic: TorchStateCritic
    """Structured policy to train."""

    optimizer: Optimizer
    """Optimizer to use"""

    loss: StateCriticCloningLoss
    """Class providing the training loss function."""

    train_stats: LogStatsAggregator = LogStatsAggregator(LogStatsLevel.EPOCH, get_stats_logger("train"))
    """Training statistics"""

    imitation_events: CriticImitationEvents = train_stats.create_event_topic(CriticImitationEvents)
    """Imitation-specific training events"""

    def train(self, n_epochs: int, eval_every_k_iterations: int = None) -> None:
        """Run training.

        :param n_epochs: How many epochs to train for
        :param eval_every_k_iterations: Number of iterations after which to run evaluation (in addition to evaluations
                                        at the end of each epoch, which are run automatically). If set to None,
                                        evaluations will run on epoch end only.
        """
        for epoch in range(n_epochs):
            # print(f"\n********** Epoch {epoch + 1} started **********")

            for iteration, data in enumerate(self.data_loader, 0):
                self._run_iteration(data)

                # Evaluate after each k iterations if set
                if eval_every_k_iterations is not None and \
                        iteration % eval_every_k_iterations == (eval_every_k_iterations - 1):
                    print(f"\n********** Critic Epoch {epoch + 1}: Iteration {iteration + 1} **********")
                    increment_log_step()

        # print(f"\n********** Final evaluation **********")
        # increment_log_step()

    def load_state_dict(self, state_dict: Dict) -> None:
        """Set the model and optimizer state.
        :param state_dict: The state dict.
        """
        self.critic.load_state_dict(state_dict)

    @override(Trainer)
    def load_state(self, file_path: Union[str, BinaryIO]) -> None:
        """implementation of :class:`~maze.train.trainers.common.trainer.Trainer`
        """
        state_dict = torch.load(file_path, map_location=torch.device(self.critic.device))
        self.load_state_dict(state_dict)

    def _run_iteration(self, data: Tuple[Dict[Union[int, str], Any], Dict[Union[int, str], Any]]) -> None:
        self.critic.train()
        self.optimizer.zero_grad()

        observation_dict, _, step_returns = data
        convert_to_torch(step_returns, device=self.critic.device, cast=None, in_place=True)

        total_loss = self.loss.calculate_loss(critic=self.critic, observation_dict=observation_dict,
                                              step_returns=step_returns, events=self.imitation_events)
        total_loss.backward()
        self.optimizer.step()

        # Report additional policy-related stats
        for policy_id in observation_dict.keys():
            l2_norm = sum([param.norm() for param in self.critic.networks[policy_id].parameters()])
            grad_norm = compute_gradient_norm(self.critic.networks[policy_id].parameters())

            self.imitation_events.critic_l2_norm(step_id=policy_id, value=l2_norm.item())
            self.imitation_events.critic_grad_norm(step_id=policy_id, value=grad_norm)
