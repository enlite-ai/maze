"""Behavioral cloning evaluation."""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.log_stats.log_stats import LogStatsLevel, LogStatsAggregator, get_stats_logger
from maze.perception.perception_utils import convert_to_torch
from maze.train.trainers.common.evaluators.evaluator import Evaluator
from maze.train.trainers.common.model_selection.model_selection_base import ModelSelectionBase
from maze.train.trainers.imitation.bc_loss import BCLoss
from maze.train.trainers.imitation.imitation_events import ImitationEvents


class BCValidationEvaluator(Evaluator):
    """Evaluates a given policy on validation data.

    Expects that the first two items returned in the dataset tuple are the observation_dict and action_dict.

    :param data_loader: The data used for evaluation.
    :param loss: Loss function to be used.
    :param model_selection: Model selection interface that will be notified of the recorded rewards.
    """

    def __init__(self,
                 loss: BCLoss,
                 model_selection: Optional[ModelSelectionBase],
                 data_loader: DataLoader,
                 logging_prefix: str = "eval"):
        self.loss = loss
        self.data_loader = data_loader
        self.model_selection = model_selection

        self.env = None
        self.eval_stats = LogStatsAggregator(LogStatsLevel.EPOCH, get_stats_logger(logging_prefix))
        self.eval_events = self.eval_stats.create_event_topic(ImitationEvents)

    @override(Evaluator)
    def evaluate(self, policy: TorchPolicy) -> None:
        """Evaluate given policy (results are stored in stat logs) and dump the model if the reward improved.

        :param policy: Policy to evaluate
        """
        policy.eval()
        with torch.no_grad():
            total_loss = []

            for iteration, data in enumerate(self.data_loader, 0):
                observation_dict, action_dict = data[:2]
                convert_to_torch(action_dict, device=policy.device, cast=None, in_place=True)

                total_loss.append(
                    self.loss.calculate_loss(policy=policy, observation_dict=observation_dict, action_dict=action_dict,
                                             events=self.eval_events).item())

            if self.model_selection:
                self.model_selection.update(-np.mean(total_loss).item())
