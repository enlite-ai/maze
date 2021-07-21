"""Runner implementations for multi-step actor critic (ACs)"""
import dataclasses
from abc import abstractmethod
from typing import Callable, Union

from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from omegaconf import DictConfig

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.env.structured_env import StructuredEnv
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.utils.factory import Factory
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.parallelization.vector_env.structured_vector_env import StructuredVectorEnv
from maze.train.parallelization.vector_env.subproc_vector_env import SubprocVectorEnv
from maze.train.trainers.common.actor_critic.actor_critic_trainer import ActorCritic
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.training_runner import TrainingRunner
from maze.utils.process import query_cpu


@dataclasses.dataclass
class ACRunner(TrainingRunner):
    """
    Abstract baseclass of AC runners.
    """

    trainer_class: Union[str, type]
    """The actor critic trainer class to be used."""
    concurrency: int
    """Number of concurrently executed environments."""
    eval_concurrency: int
    """ Number of concurrent evaluation envs """

    def __post_init__(self):
        """
        Adjusts initial values where necessary.
        """

        if self.concurrency <= 0:
            self.concurrency = query_cpu()
        if self.eval_concurrency <= 0:
            self.eval_concurrency = query_cpu()

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

        # initialize actor critic model
        model = TorchActorCritic(
            policy=self._model_composer.policy,
            critic=self._model_composer.critic,
            device=cfg.algorithm.device)

        # initialize best model selection
        self._model_selection = BestModelSelection(dump_file=self.state_dict_dump_file, model=model,
                                                   dump_interval=self.dump_interval)

        # initialize the env and enable statistics collection
        evaluator = None
        if cfg.algorithm.rollout_evaluator.n_episodes > 0:
            eval_env = self.create_distributed_env(self.env_factory, self.eval_concurrency, logging_prefix="eval")
            eval_env_instance_seeds = [self.maze_seeding.generate_env_instance_seed()
                                       for _ in range(self.eval_concurrency)]
            eval_env.seed(eval_env_instance_seeds)

            # initialize rollout evaluator
            evaluator = Factory(base_type=RolloutEvaluator).instantiate(cfg.algorithm.rollout_evaluator,
                                                                        eval_env=eval_env,
                                                                        model_selection=self._model_selection)

        # look up model class
        trainer_class = Factory(base_type=ActorCritic).type_from_name(self.trainer_class)

        # initialize trainer (from input directory)
        self._trainer = trainer_class(
            algorithm_config=cfg.algorithm,
            rollout_generator=RolloutGenerator(env=envs),
            evaluator=evaluator,
            model=model,
            model_selection=self._model_selection
        )

        self._init_trainer_from_input_dir(trainer=self._trainer, state_dict_dump_file=self.state_dict_dump_file,
                                          input_dir=cfg.input_dir)

    @abstractmethod
    def create_distributed_env(self,
                               env_factory: Callable[[], Union[MazeEnv, StructuredEnv]],
                               concurrency: int,
                               logging_prefix: str
                               ) -> StructuredVectorEnv:
        """The dev and local runner implement the setup of the distribution env"""


@dataclasses.dataclass
class ACDevRunner(ACRunner):
    """Runner for single-threaded training, based on SequentialVectorEnv."""

    def create_distributed_env(self,
                               env_factory: Callable[[], Union[MazeEnv, StructuredEnv]],
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
                               env_factory: Callable[[], Union[MazeEnv, StructuredEnv]],
                               concurrency: int,
                               logging_prefix: str
                               ) -> SubprocVectorEnv:
        """create multi-process env distribution"""
        return SubprocVectorEnv([env_factory for _ in range(concurrency)], logging_prefix=logging_prefix)
