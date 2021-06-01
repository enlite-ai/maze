"""Runner implementations for Behavioral Cloning."""
from abc import abstractmethod
import dataclasses
from typing import Tuple, Callable, Union, Optional, List

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
from maze.train.trainers.common.evaluators.evaluator import Evaluator
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


@dataclasses.dataclass
class BCRunner(TrainingRunner):
    """
    Dev runner for imitation learning.
    Loads the given trajectory data and trains a policy on top of it using supervised learning.
    """

    dataset: DictConfig
    """Specify the Dataset class used to load the trajectory data for training."""
    eval_concurrency: int
    """Number of concurrent evaluation envs."""

    evaluators: Optional[List[BCValidationEvaluator]] = dataclasses.field(default=None, init=False)

    @override(TrainingRunner)
    def setup(self, cfg: DictConfig) -> None:
        """
        See :py:meth:`~maze.train.trainers.common.training_runner.TrainingRunner.setup`.
        """

        super().setup(cfg)

        env = self.env_factory()

        with SwitchWorkingDirectoryToInput(cfg.input_dir):
            dataset = Factory(base_type=Dataset).instantiate(self.dataset, conversion_env_factory=self.env_factory)

        assert len(dataset) > 0, f"Expected to find trajectory data, but did not find any. Please check that " \
                                 f"the path you supplied is correct."
        validation, train = self._split_dataset(dataset, cfg.algorithm.validation_percentage,
                                                self.maze_seeding.generate_env_instance_seed())

        # Create data loaders
        torch_generator = torch.Generator().manual_seed(self.maze_seeding.generate_env_instance_seed())
        train_data_loader = DataLoader(train, shuffle=True, batch_size=cfg.algorithm.batch_size,
                                       generator=torch_generator)
        validation_data_loader = DataLoader(validation, batch_size=cfg.algorithm.batch_size,
                                            generator=torch_generator)

        policy = TorchPolicy(networks=self._model_composer.policy.networks,
                             distribution_mapper=self._model_composer.distribution_mapper, device=cfg.algorithm.device)
        policy.seed(self.maze_seeding.agent_global_seed)

        self._model_selection = BestModelSelection(self.state_dict_dump_file, policy)
        optimizer = Factory(Optimizer).instantiate(cfg.algorithm.optimizer, params=policy.parameters())
        loss = BCLoss(action_spaces_dict=env.action_spaces_dict, entropy_coef=cfg.algorithm.entropy_coef)

        self._trainer = BCTrainer(
            algorithm_config=self._cfg.algorithm,
            data_loader=train_data_loader,
            policy=policy,
            optimizer=optimizer,
            loss=loss)

        # initialize model from input_dir
        self._init_trainer_from_input_dir(
            trainer=self._trainer, state_dict_dump_file=self.state_dict_dump_file, input_dir=cfg.input_dir
        )

        # evaluate using the validation set
        self.evaluators = [BCValidationEvaluator(
            data_loader=validation_data_loader, loss=loss, logging_prefix="eval-validation",
            model_selection=self._model_selection  # use the validation set evaluation to select the best model
        )]

        # if evaluation episodes are set, perform additional evaluation by policy rollout
        if cfg.algorithm.n_eval_episodes > 0:
            eval_env = self.create_distributed_eval_env(self.env_factory, self.eval_concurrency,
                                                        logging_prefix="eval-rollout")
            eval_env_instance_seeds = [self.maze_seeding.generate_env_instance_seed() for _ in
                                       range(self.eval_concurrency)]
            eval_env.seed(eval_env_instance_seeds)
            self.evaluators += [
                RolloutEvaluator(eval_env, n_episodes=cfg.algorithm.n_eval_episodes, model_selection=None)
            ]

    @override(TrainingRunner)
    def run(
        self,
        n_epochs: Optional[int] = None,
        evaluator: Optional[Evaluator] = None,
        eval_every_k_iterations: Optional[int] = None
    ) -> None:
        """
        Run the training master node.
        See :py:meth:`~maze.train.trainers.common.training_runner.TrainingRunner.run`.
        :param evaluator: Evaluator to use for evaluation rollouts
        :param n_epochs: How many epochs to train for
        :param eval_every_k_iterations: Number of iterations after which to run evaluation (in addition to evaluations
        at the end of each epoch, which are run automatically). If set to None, evaluations will run on epoch end only.
        """

        self._trainer.train(
            n_epochs=self._cfg.algorithm.n_epochs if n_epochs is None else n_epochs,
            eval_every_k_iterations=(
                self._cfg.algorithm.eval_every_k_iterations
                if eval_every_k_iterations is None else eval_every_k_iterations
            ),
            evaluator=MultiEvaluator(self.evaluators) if evaluator is None else evaluator
        )

    @staticmethod
    def _split_dataset(dataset: Dataset, validation_percentage: float,
                       env_seed: int) -> Tuple[Subset, Subset]:
        """
        Split the given dataset into validation and training set based on the runner configuration.

        :param dataset: Dataset to split.
        :param env_seed: Seed for splitting the dataset.

        :return: Tuple of subsets: (validation, train).
        """
        validation_size = int(np.round(validation_percentage * len(dataset) / 100))

        method = getattr(dataset, 'random_split', None)
        if method is not None and callable(method):
            return method([validation_size, len(dataset) - validation_size], torch.Generator().manual_seed(1234))
        else:
            return torch.utils.data.random_split(
                dataset=dataset,
                lengths=[validation_size, len(dataset) - validation_size],
                generator=torch.Generator().manual_seed(env_seed))

    @classmethod
    @abstractmethod
    def create_distributed_eval_env(cls,
                                    env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> VectorEnv:
        """The individual runners implement the setup of the distributed eval env"""


@dataclasses.dataclass
class BCDevRunner(BCRunner):
    """Runner for single-threaded training, based on SequentialVectorEnv."""

    @classmethod
    @override(BCRunner)
    def create_distributed_eval_env(
        cls,
        env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
        eval_concurrency: int,
        logging_prefix: str
    ) -> SequentialVectorEnv:
        """create single-threaded env distribution"""
        return SequentialVectorEnv([env_factory for _ in range(eval_concurrency)], logging_prefix=logging_prefix)


@dataclasses.dataclass
class BCLocalRunner(BCRunner):
    """Runner for locally distributed training, based on SubprocVectorEnv."""

    @classmethod
    @override(BCRunner)
    def create_distributed_eval_env(
        cls,
        env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
        eval_concurrency: int,
        logging_prefix: str
    ) -> SubprocVectorEnv:
        """create multi-process env distribution"""
        return SubprocVectorEnv([env_factory for _ in range(eval_concurrency)], logging_prefix=logging_prefix)
