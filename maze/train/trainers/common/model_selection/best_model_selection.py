"""Contains a best model selection."""

from typing import Optional

import numpy as np
import torch

from maze.core.agent.torch_model import TorchModel
from maze.core.annotations import override
from maze.train.trainers.common.model_selection.model_selection_base import ModelSelectionBase
from maze.utils.bcolors import BColors


class BestModelSelection(ModelSelectionBase):
    """Best model selection strategy.

    :param dump_file: Specifies the file path to dump the policy state for the best reward.
    :param model: The model to be dumped.
    """

    def __init__(self, dump_file: Optional[str], model: Optional[TorchModel]):
        self.dump_file = dump_file
        self.model = model

        self.last_improvement = 0
        self.best_reward = -np.inf

    @override(ModelSelectionBase)
    def update(self, reward: float) -> None:
        """Implementation of ModelSelection.update().
        """
        self.last_improvement += 1
        if reward > self.best_reward:
            BColors.print_colored("-> new overall best model {:.5f}!".format(reward), color=BColors.OKBLUE)
            self.best_reward = reward
            self.last_improvement = 0

            # save the state to a file
            if self.dump_file:
                state_dict = self.model.state_dict()
                torch.save(state_dict, self.dump_file)
