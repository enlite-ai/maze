"""Runner implementations for multi-step IMPALA"""
import copy
from abc import abstractmethod
from dataclasses import dataclass
from typing import Union, Callable

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.train.parallelization.distributed_actors.distributed_actors import BaseDistributedActors
from maze.train.parallelization.distributed_actors.dummy_distributed_actors import DummyDistributedActors
from maze.train.parallelization.distributed_actors.subproc_distributed_actors import SubprocDistributedActors
from maze.train.parallelization.distributed_env.distributed_env import BaseDistributedEnv
from maze.train.parallelization.distributed_env.dummy_distributed_env import DummyStructuredDistributedEnv
from maze.train.parallelization.distributed_env.subproc_distributed_env import SubprocStructuredDistributedEnv
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.training_runner import TrainingRunner
from maze.train.trainers.impala.impala_trainer import MultiStepIMPALA
from omegaconf import DictConfig


@dataclass
class ImpalaRunner(TrainingRunner):
    """Runner for single-threaded training, based on DummyStructuredDistributedEnv."""

    @override(TrainingRunner)
    def run(self, cfg: DictConfig) -> None:
        """Run local in-thread training."""
        super().run(cfg)

        # initialize actor critic model
        model = TorchActorCritic(
            policy=TorchPolicy(networks=self.model_composer.policy.networks,
                               distribution_mapper=self.model_composer.distribution_mapper,
                               device=cfg.algorithm.device),
            critic=self.model_composer.critic,
            device=cfg.algorithm.device)

        # policy for distributed actor rollouts
        #  - does not need critic
        #  - distributed actors are supported on CPU only
        #  - during training, new policy versions will be distributed to actors (as the training policy is updated)
        rollout_policy = copy.deepcopy(model.policy)
        rollout_policy.to("cpu")

        # initialize the env and enable statistics collection
        eval_env = self.create_distributed_eval_env(self.env_factory,
                                                    cfg.algorithm.eval_concurrency,
                                                    logging_prefix="eval")
        rollout_actors = self.create_distributed_rollout_actors(self.env_factory, rollout_policy,
                                                                cfg.algorithm.n_rollout_steps,
                                                                cfg.algorithm.num_actors,
                                                                cfg.algorithm.actors_batch_size,
                                                                cfg.algorithm.queue_out_of_sync_factor)

        # initialize optimizer
        trainer = MultiStepIMPALA(model=model, rollout_actors=rollout_actors, eval_env=eval_env,
                                  options=cfg.algorithm)

        # initialize model from input_dir
        self._init_trainer_from_input_dir(trainer=trainer, state_dict_dump_file=self.state_dict_dump_file,
                                          input_dir=cfg.input_dir)

        # initialize best model selection
        model_selection = BestModelSelection(dump_file=self.state_dict_dump_file, model=model)

        # train agent
        trainer.train(n_epochs=cfg.algorithm.n_epochs,
                      epoch_length=cfg.algorithm.epoch_length,
                      deterministic_eval=cfg.algorithm.deterministic_eval,
                      eval_repeats=cfg.algorithm.eval_repeats,
                      patience=cfg.algorithm.patience,
                      model_selection=model_selection)

    @abstractmethod
    def create_distributed_eval_env(self,
                                    env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> BaseDistributedEnv:
        """The individual runners implement the setup of the distributed eval env"""

    @abstractmethod
    def create_distributed_rollout_actors(
            self,
            env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
            policy: TorchPolicy,
            n_rollout_steps: int,
            n_actors: int,
            batch_size: int,
            queue_out_of_sync_factor: float) -> BaseDistributedActors:
        """The individual runners implement the setup of the distributed training rollout actors"""


@dataclass
class ImpalaDevRunner(ImpalaRunner):
    """Runner for single-threaded training, based on DummyStructuredDistributedEnv."""

    def create_distributed_rollout_actors(
            self,
            env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
            policy: TorchPolicy,
            n_rollout_steps: int,
            n_actors: int,
            batch_size: int,
            queue_out_of_sync_factor: float) -> DummyDistributedActors:
        """Create dummy (sequentially-executed) actors."""
        return DummyDistributedActors(env_factory, policy, n_rollout_steps, n_actors, batch_size)

    def create_distributed_eval_env(self,
                                    env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> DummyStructuredDistributedEnv:
        """create single-threaded env distribution"""
        # fallback to a fixed number of pseudo-concurrent environments to avoid making this sequential execution
        # unnecessary slow on machines with a higher core number
        return DummyStructuredDistributedEnv([env_factory for _ in range(eval_concurrency)],
                                             logging_prefix=logging_prefix)


@dataclass
class ImpalaLocalRunner(ImpalaRunner):
    """Runner for locally distributed training, based on SubprocStructuredDistributedEnv."""

    start_method: str
    """Type of start method used for multiprocessing ('forkserver', 'spawn', 'fork')"""

    def create_distributed_rollout_actors(
            self,
            env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
            policy: TorchPolicy,
            n_rollout_steps: int,
            n_actors: int,
            batch_size: int,
            queue_out_of_sync_factor: float) -> SubprocDistributedActors:
        """Create dummy (sequentially-executed) actors."""
        return SubprocDistributedActors(env_factory, policy, n_rollout_steps, n_actors, batch_size,
                                        queue_out_of_sync_factor, self.start_method)

    def create_distributed_eval_env(self,
                                    env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> SubprocStructuredDistributedEnv:
        """create multi-process env distribution"""
        return SubprocStructuredDistributedEnv([env_factory for _ in range(eval_concurrency)],
                                               logging_prefix=logging_prefix)


