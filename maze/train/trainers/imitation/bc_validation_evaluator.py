"""Behavioral cloning evaluation."""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.structured_env import ActorID
from maze.core.log_stats.log_stats import LogStatsLevel, LogStatsAggregator, get_stats_logger
from maze.perception.perception_utils import convert_to_torch
from maze.train.trainers.common.evaluators.evaluator import Evaluator
from maze.train.trainers.common.model_selection.model_selection_base import ModelSelectionBase
from maze.train.trainers.imitation.bc_loss import BCLoss
from maze.train.trainers.imitation.imitation_events import ImitationEvents
from maze.train.utils.train_utils import debatch_actor_ids


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
                 logging_prefix: Optional[str] = "eval"):
        self.loss = loss
        self.data_loader = data_loader
        self.model_selection = model_selection

        self.env = None
        if logging_prefix:
            self.eval_stats = LogStatsAggregator(LogStatsLevel.EPOCH, get_stats_logger(logging_prefix))
        else:
            self.eval_stats = LogStatsAggregator(LogStatsLevel.EPOCH)
        self.eval_events = self.eval_stats.create_event_topic(ImitationEvents)

    @override(Evaluator)
    def evaluate(self, policy: TorchPolicy) -> None:
        """Evaluate given policy (results are stored in stat logs) and dump the model if the reward improved.

        :param policy: Policy to evaluate.
        """
        policy.eval()
        with torch.no_grad():
            total_loss = []

            for iteration, data in enumerate(self.data_loader, 0):
                observations, actions, actor_ids = data[0], data[1], data[-1]
                action_logits = None if len(data) == 3 else data[2]
                actor_ids = debatch_actor_ids(actor_ids)
                # Convert only actions to torch, since observations are converted in
                # policy.compute_substep_policy_output method
                convert_to_torch(actions, device=policy.device, cast=None, in_place=True)
                if action_logits is not None:
                    convert_to_torch(action_logits, device=policy.device, cast=None, in_place=True)

                total_loss.append(
                    self.loss.calculate_loss(policy=policy, observations=observations, actions=actions,
                                             events=self.eval_events, actor_ids=actor_ids, action_logits=action_logits).item())

            if self.model_selection:
                self.model_selection.update(-np.mean(total_loss).item())
