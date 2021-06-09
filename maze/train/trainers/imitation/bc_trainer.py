"""Trainer class for behavioral cloning."""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Union, Optional

import torch
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.log_stats.log_stats import LogStatsAggregator, LogStatsLevel, get_stats_logger, increment_log_step
from maze.perception.perception_utils import convert_to_torch
from maze.train.trainers.common.trainer import Trainer
from maze.train.trainers.imitation.bc_algorithm_config import BCAlgorithmConfig
from maze.train.trainers.imitation.bc_loss import BCLoss
from maze.train.trainers.common.evaluators.evaluator import Evaluator
from maze.train.trainers.imitation.imitation_events import ImitationEvents
from maze.train.utils.train_utils import compute_gradient_norm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing.io import BinaryIO


@dataclass
class BCTrainer(Trainer):
    """Trainer for behavioral cloning learning.

    Runs training on top of provided trajectory data and rolls out the policy using the provided evaluator.

    In structured (multi-step) envs, all policies are trained simultaneously based on the substep actions
    and observation present in the trajectory data.
    """

    data_loader: DataLoader
    """Data loader for loading trajectory data."""

    policy: TorchPolicy
    """Structured policy to train."""

    optimizer: Optimizer
    """Optimizer to use"""

    loss: BCLoss
    """Class providing the training loss function."""

    train_stats: LogStatsAggregator = LogStatsAggregator(LogStatsLevel.EPOCH, get_stats_logger("train"))
    """Training statistics"""

    imitation_events: ImitationEvents = train_stats.create_event_topic(ImitationEvents)
    """Imitation-specific training events"""

    def __init__(
        self,
        algorithm_config: BCAlgorithmConfig,
        data_loader: DataLoader,
        policy: TorchPolicy,
        optimizer: Optimizer,
        loss: BCLoss
    ):
        super().__init__(algorithm_config)

        self.data_loader = data_loader
        self.policy = policy
        self.optimizer = optimizer
        self.loss = loss

    @override(Trainer)
    def train(
        self, evaluator: Evaluator, n_epochs: Optional[int] = None, eval_every_k_iterations: Optional[int] = None
    ) -> None:
        """
        Run training.
        :param evaluator: Evaluator to use for evaluation rollouts
        :param n_epochs: How many epochs to train for
        :param eval_every_k_iterations: Number of iterations after which to run evaluation (in addition to evaluations
        at the end of each epoch, which are run automatically). If set to None, evaluations will run on epoch end only.
        """

        if n_epochs is None:
            n_epochs = self.algorithm_config.n_epochs
        if eval_every_k_iterations is None:
            eval_every_k_iterations = self.algorithm_config.eval_every_k_iterations

        for epoch in range(n_epochs):
            print(f"\n********** Epoch {epoch + 1} started **********")
            evaluator.evaluate(self.policy)
            increment_log_step()

            for iteration, data in enumerate(self.data_loader, 0):
                self._run_iteration(data)

                # Evaluate after each k iterations if set
                if eval_every_k_iterations is not None and \
                        iteration % eval_every_k_iterations == (eval_every_k_iterations - 1):
                    print(f"\n********** Epoch {epoch + 1}: Iteration {iteration + 1} **********")
                    evaluator.evaluate(self.policy)
                    increment_log_step()

        print(f"\n********** Final evaluation **********")
        evaluator.evaluate(self.policy)
        increment_log_step()

    def load_state_dict(self, state_dict: Dict) -> None:
        """Set the model and optimizer state.
        :param state_dict: The state dict.
        """
        self.policy.load_state_dict(state_dict)

    @override(Trainer)
    def load_state(self, file_path: Union[str, BinaryIO]) -> None:
        """implementation of :class:`~maze.train.trainers.common.trainer.Trainer`
        """
        state_dict = torch.load(file_path, map_location=torch.device(self.policy.device))
        self.load_state_dict(state_dict)

    def _run_iteration(self, data: Tuple[Dict[Union[int, str], Any], Dict[Union[int, str], Any]]) -> None:
        self.policy.train()
        self.optimizer.zero_grad()

        observation_dict, action_dict = data
        convert_to_torch(action_dict, device=self.policy.device, cast=None, in_place=True)

        total_loss = self.loss.calculate_loss(policy=self.policy, observation_dict=observation_dict,
                                              action_dict=action_dict, events=self.imitation_events)
        total_loss.backward()
        self.optimizer.step()

        # Report additional policy-related stats
        for policy_id in observation_dict.keys():
            l2_norm = sum([param.norm() for param in self.policy.networks[policy_id].parameters()])
            grad_norm = compute_gradient_norm(self.policy.networks[policy_id].parameters())

            self.imitation_events.policy_l2_norm(step_id=policy_id, value=l2_norm.item())
            self.imitation_events.policy_grad_norm(step_id=policy_id, value=grad_norm)
