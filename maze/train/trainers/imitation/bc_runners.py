"""Runner implementations for Behavioral Cloning."""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, Callable, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.utils.config_utils import SwitchWorkingDirectoryToInput
from maze.core.utils.factory import Factory
from maze.train.trainers.common.evaluators.multi_evaluator import MultiEvaluator
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.parallelization.vector_env.vector_env import VectorEnv
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.parallelization.vector_env.subproc_vector_env import SubprocVectorEnv
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.training_runner import TrainingRunner
from maze.train.trainers.imitation.bc_loss import BCLoss
from maze.train.trainers.imitation.bc_trainer import BCTrainer
from maze.train.trainers.imitation.bc_validation_evaluator import BCValidationEvaluator


@dataclass
class BCRunner(TrainingRunner):
    """Dev runner for imitation learning.

    Loads the given trajectory data and trains a policy on top of it using supervised learning.
    """
    dataset: DictConfig
    """Specify the Dataset class used to load the trajectory data for training"""

    eval_concurrency: int
    """Number of concurrent evaluation envs"""

    @override(TrainingRunner)
    def run(self, cfg: DictConfig) -> None:
        """Run the training master node."""
        super().run(cfg)
        env = self.env_factory()

        with SwitchWorkingDirectoryToInput(cfg.input_dir):
            dataset = Factory(base_type=Dataset).instantiate(self.dataset, conversion_env_factory=self.env_factory)

        assert len(dataset) > 0, f"Expected to find trajectory data, but did not find any. Please check that " \
                                 f"the path you supplied is correct."
        validation, train = self._split_dataset(dataset, cfg.algorithm.validation_percentage)

        # Create data loaders
        train_data_loader = DataLoader(train, shuffle=True, batch_size=cfg.algorithm.batch_size)
        validation_data_loader = DataLoader(validation, batch_size=cfg.algorithm.batch_size)

        policy = TorchPolicy(networks=self.model_composer.policy.networks,
                             distribution_mapper=self.model_composer.distribution_mapper, device=cfg.algorithm.device)

        model_selection = BestModelSelection(self.state_dict_dump_file, policy)
        optimizer = Factory(Optimizer).instantiate(cfg.algorithm.optimizer, params=policy.parameters())
        loss = BCLoss(action_spaces_dict=env.action_spaces_dict, entropy_coef=cfg.algorithm.entropy_coef)

        trainer = BCTrainer(
            data_loader=train_data_loader,
            policy=policy,
            optimizer=optimizer,
            loss=loss)

        # initialize model from input_dir
        self._init_trainer_from_input_dir(trainer=trainer, state_dict_dump_file=self.state_dict_dump_file,
                                          input_dir=cfg.input_dir)

        # evaluate using the validation set
        evaluators = [BCValidationEvaluator(
            data_loader=validation_data_loader, loss=loss, logging_prefix="eval-validation",
            model_selection=model_selection  # use the validation set evaluation to select the best model
        )]

        # if evaluation episodes are set, perform additional evaluation by policy rollout
        if cfg.algorithm.n_eval_episodes > 0:
            eval_env = self.create_distributed_eval_env(self.env_factory, self.eval_concurrency,
                                                        logging_prefix="eval-rollout")
            evaluators += [RolloutEvaluator(eval_env, n_episodes=cfg.algorithm.n_eval_episodes, model_selection=None)]

        trainer.train(
            n_epochs=cfg.algorithm.n_epochs,
            evaluator=MultiEvaluator(evaluators),
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

    @abstractmethod
    def create_distributed_eval_env(self,
                                    env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> VectorEnv:
        """The individual runners implement the setup of the distributed eval env"""


@dataclass
class BCDevRunner(BCRunner):
    """Runner for single-threaded training, based on SequentialVectorEnv."""

    def create_distributed_eval_env(self,
                                    env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> SequentialVectorEnv:
        """create single-threaded env distribution"""
        return SequentialVectorEnv([env_factory for _ in range(eval_concurrency)],
                                   logging_prefix=logging_prefix)


@dataclass
class BCLocalRunner(BCRunner):
    """Runner for locally distributed training, based on SubprocVectorEnv."""

    def create_distributed_eval_env(self,
                                    env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> SubprocVectorEnv:
        """create multi-process env distribution"""
        return SubprocVectorEnv([env_factory for _ in range(eval_concurrency)],
                                logging_prefix=logging_prefix)
