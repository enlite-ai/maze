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
    :param dump_interval: Update count interval between regularly dumping the model parameters.
    """

    def __init__(self, dump_file: Optional[str], model: Optional[TorchModel], dump_interval: Optional[int]):
        self.dump_file = dump_file
        self.model = model
        self.dump_interval = dump_interval

        self.last_improvement = 0
        self.best_reward = -np.inf
        self.update_count = 0

    @override(ModelSelectionBase)
    def update(self, reward: float) -> None:
        """Implementation of ModelSelection.update().

        :param reward: Reward (score) used for best model selection.
        """
        self.last_improvement += 1
        if reward > self.best_reward:
            BColors.print_colored(f"-> new overall best model {reward:.5f}!", color=BColors.OKBLUE)
            self.best_reward = reward
            self.last_improvement = 0

            # save the state to a file
            if self.dump_file:
                BColors.print_colored(f"-> dumping model to {self.dump_file}!", color=BColors.OKBLUE)
                state_dict = self.model.state_dict()
                torch.save(state_dict, self.dump_file)

            # regularly dump model
            if self.dump_interval and self.update_count % self.dump_interval:
                dump_file = self.dump_file.replace('.pt', f'-epoch_{self.update_count}.pt')
                BColors.print_colored(f"-> dumping model to {dump_file}!", color=BColors.OKBLUE)
                state_dict = self.model.state_dict()
                torch.save(state_dict, dump_file)
            self.update_count += 1
