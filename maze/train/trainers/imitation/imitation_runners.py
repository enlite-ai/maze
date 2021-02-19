"""Runner implementations for Imitation Learning"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.utils.registry import Registry
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.training_runner import TrainingRunner
from maze.train.trainers.imitation.bc_evaluator import BCEvaluator
from maze.train.trainers.imitation.bc_loss import BCLoss
from maze.train.trainers.imitation.bc_trainer import BCTrainer


@dataclass
class ImitationRunner(TrainingRunner):
    """Dev runner for imitation learning.

    Loads the given trajectory data and trains a policy on top of it using supervised learning.
    """
    dataset: DictConfig
    """Specify the Dataset class used to load the trajectory data for training"""

    @override(TrainingRunner)
    def run(self, cfg: DictConfig) -> None:
        """Run the training master node."""
        super().run(cfg)
        env = self.env_factory()

        dataset = Registry(base_type=Dataset).arg_to_obj(self.dataset, env_factory=self.env_factory)
        assert len(dataset) > 0, f"Expected to find trajectory data, but did not find any. Please check that " \
                                 f"the path you supplied is correct."
        validation, train = self._split_dataset(dataset, cfg.algorithm.validation_percentage)

        # Create data loaders
        train_data_loader = DataLoader(train, shuffle=True, batch_size=cfg.algorithm.batch_size)
        validation_data_loader = DataLoader(validation, batch_size=cfg.algorithm.batch_size)

        policy = TorchPolicy(networks=self.model_composer.policy.networks,
                             distribution_mapper=self.model_composer.distribution_mapper,
                             device=cfg.algorithm.device)

        model_selection = BestModelSelection(self.state_dict_dump_file, policy)
        optimizer = Registry(Optimizer).arg_to_obj(cfg.algorithm.optimizer, params=policy.parameters())
        loss = BCLoss(action_spaces_dict=env.action_spaces_dict)

        trainer = BCTrainer(
            data_loader=train_data_loader,
            policy=policy,
            optimizer=optimizer,
            loss=loss)

        # initialize model from input_dir
        self._init_trainer_from_input_dir(trainer=trainer, state_dict_dump_file=self.state_dict_dump_file,
                                          input_dir=cfg.input_dir)

        evaluator = BCEvaluator(
            data_loader=validation_data_loader,
            loss=loss,
            model_selection=model_selection)

        trainer.train(
            n_epochs=cfg.algorithm.n_epochs,
            evaluator=evaluator,
            eval_every_k_iterations=cfg.algorithm.eval_every_k_iterations)

    @staticmethod
    def _split_dataset(dataset: Dataset, validation_percentage: float) -> Tuple[Subset, Subset]:
        """
        Split the given dataset into validation and training set based on the runner configuration.

        :param dataset: Dataset to split
        :return: Tuple of subsets: (validation, train)
        """
        validation_size = int(np.round(validation_percentage * len(dataset) / 100))

        method = getattr(dataset, 'random_split', None)
        if method is not None and callable(method):
            return method([validation_size, len(dataset) - validation_size], torch.Generator().manual_seed(1234))
        else:
            return torch.utils.data.random_split(
                dataset=dataset,
                lengths=[validation_size, len(dataset) - validation_size],
                generator=torch.Generator().manual_seed(1234))
