"""Runner implementations for multi-step actor critic (ACs)"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Union

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.utils.registry import Registry
from maze.train.parallelization.distributed_env.distributed_env import BaseDistributedEnv
from maze.train.parallelization.distributed_env.dummy_distributed_env import DummyStructuredDistributedEnv
from maze.train.parallelization.distributed_env.subproc_distributed_env import SubprocStructuredDistributedEnv
from maze.train.trainers.common.actor_critic.actor_critic_trainer import MultiStepActorCritic
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.training_runner import TrainingRunner
from omegaconf import DictConfig


@dataclass
class ACRunner(TrainingRunner):
    """Abstract baseclass of AC runners."""

    trainer_class: Union[str, type]
    """The actor critic trainer class to be used"""

    concurrency: int
    """Number of concurrently executed environments."""

    initial_state_dict: str
    """path to initial state (policy weights, critic weights, optimizer state)"""

    @override(TrainingRunner)
    def run(self, cfg: DictConfig) -> None:
        """Run training."""
        super().run(cfg)

        # initialize distributed env
        envs = self.create_distributed_env(self.env_factory, self.concurrency, logging_prefix="train")

        # initialize the env and enable statistics collection
        eval_env = self.create_distributed_env(self.env_factory, self.concurrency, logging_prefix="eval")

        # initialize actor critic model
        model = TorchActorCritic(
            policy=self.model_composer.policy,
            critic=self.model_composer.critic,
            device=cfg.algorithm.device)

        # initialize best model selection
        model_selection = BestModelSelection(dump_file=self.state_dict_dump_file, model=model)

        # look up model class
        trainer_class = Registry(base_type=MultiStepActorCritic).class_type_from_module_name(self.trainer_class)

        # initialize trainer (from input directory)
        trainer = trainer_class(algorithm_config=cfg.algorithm, env=envs, eval_env=eval_env, model=model,
                                model_selection=model_selection, initial_state=self.initial_state_dict)

        self._init_trainer_from_input_dir(trainer=trainer, state_dict_dump_file=self.state_dict_dump_file,
                                          input_dir=cfg.input_dir)

        # start training
        trainer.train()

    @abstractmethod
    def create_distributed_env(self,
                               env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                               concurrency: int,
                               logging_prefix: str
                               ) -> BaseDistributedEnv:
        """The dev and local runner implement the setup of the distribution env"""


class ACDevRunner(ACRunner):
    """Runner for single-threaded training, based on DummyStructuredDistributedEnv."""

    def create_distributed_env(self,
                               env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                               concurrency: int,
                               logging_prefix: str
                               ) -> DummyStructuredDistributedEnv:
        """create single-threaded env distribution"""
        # fallback to a fixed number of pseudo-concurrent environments to avoid making this sequential execution
        # unnecessary slow on machines with a higher core number
        return DummyStructuredDistributedEnv([env_factory for _ in range(concurrency)], logging_prefix=logging_prefix)


class ACLocalRunner(ACRunner):
    """Runner for locally distributed training, based on DummyStructuredDistributedEnv."""

    def create_distributed_env(self,
                               env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                               concurrency: int,
                               logging_prefix: str
                               ) -> SubprocStructuredDistributedEnv:
        """create multi-process env distribution"""
        return SubprocStructuredDistributedEnv([env_factory for _ in range(concurrency)], logging_prefix=logging_prefix)


