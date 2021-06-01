"""Runner implementations for multi-step actor critic (ACs)"""
from abc import abstractmethod
import dataclasses
from typing import Callable, Union

from omegaconf import DictConfig

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.utils.factory import Factory
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.parallelization.vector_env.subproc_vector_env import SubprocVectorEnv
from maze.train.parallelization.vector_env.vector_env import VectorEnv
from maze.train.trainers.common.actor_critic.actor_critic_trainer import ActorCritic
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.training_runner import TrainingRunner


@dataclasses.dataclass
class ACRunner(TrainingRunner):
    """
    Abstract baseclass of AC runners.
    """

    trainer_class: Union[str, type]
    """The actor critic trainer class to be used."""
    concurrency: int
    """Number of concurrently executed environments."""
    initial_state_dict: str
    """Path to initial state (policy weights, critic weights, optimizer state)."""

    @override(TrainingRunner)
    def setup(self, cfg: DictConfig) -> None:
        """
        See :py:meth:`~maze.train.trainers.common.training_runner.TrainingRunner.setup`.
        """

        super().setup(cfg)

        # initialize distributed env
        envs = self.create_distributed_env(self.env_factory, self.concurrency, logging_prefix="train")
        train_env_instance_seeds = [self.maze_seeding.generate_env_instance_seed() for _ in range(self.concurrency)]
        envs.seed(train_env_instance_seeds)

        # initialize the env and enable statistics collection
        eval_env = None
        if cfg.algorithm.eval_repeats > 0:
            eval_env = self.create_distributed_env(self.env_factory, self.concurrency, logging_prefix="eval")
            eval_env_instance_seeds = [self.maze_seeding.generate_env_instance_seed() for _ in range(self.concurrency)]
            eval_env.seed(eval_env_instance_seeds)
        # initialize actor critic model
        model = TorchActorCritic(
            policy=self._model_composer.policy,
            critic=self._model_composer.critic,
            device=cfg.algorithm.device)

        # initialize best model selection
        self._model_selection = BestModelSelection(dump_file=self.state_dict_dump_file, model=model)

        # look up model class
        trainer_class = Factory(base_type=ActorCritic).type_from_name(self.trainer_class)

        # initialize trainer (from input directory)
        self._trainer = trainer_class(
            algorithm_config=cfg.algorithm,
            env=envs,
            eval_env=eval_env,
            model=model,
            model_selection=self._model_selection,
            initial_state=self.initial_state_dict
        )

        self._init_trainer_from_input_dir(trainer=self._trainer, state_dict_dump_file=self.state_dict_dump_file,
                                          input_dir=cfg.input_dir)

    @abstractmethod
    def create_distributed_env(self,
                               env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                               concurrency: int,
                               logging_prefix: str
                               ) -> VectorEnv:
        """The dev and local runner implement the setup of the distribution env"""


@dataclasses.dataclass
class ACDevRunner(ACRunner):
    """Runner for single-threaded training, based on SequentialVectorEnv."""

    def create_distributed_env(self,
                               env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                               concurrency: int,
                               logging_prefix: str
                               ) -> SequentialVectorEnv:
        """create single-threaded env distribution"""
        # fallback to a fixed number of pseudo-concurrent environments to avoid making this sequential execution
        # unnecessary slow on machines with a higher core number
        return SequentialVectorEnv([env_factory for _ in range(concurrency)], logging_prefix=logging_prefix)


@dataclasses.dataclass
class ACLocalRunner(ACRunner):
    """Runner for locally distributed training, based on SubprocVectorEnv."""

    def create_distributed_env(self,
                               env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                               concurrency: int,
                               logging_prefix: str
                               ) -> SubprocVectorEnv:
        """create multi-process env distribution"""
        return SubprocVectorEnv([env_factory for _ in range(concurrency)], logging_prefix=logging_prefix)
