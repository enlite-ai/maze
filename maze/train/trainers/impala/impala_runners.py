"""Runner implementations for multi-step IMPALA"""
import copy
from abc import abstractmethod
import dataclasses
from typing import Union, Callable, Optional, List

from omegaconf import DictConfig

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.train.parallelization.distributed_actors.distributed_actors import DistributedActors
from maze.train.parallelization.distributed_actors.sequential_distributed_actors import SequentialDistributedActors
from maze.train.parallelization.distributed_actors.subproc_distributed_actors import SubprocDistributedActors
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.parallelization.vector_env.subproc_vector_env import SubprocVectorEnv
from maze.train.parallelization.vector_env.vector_env import VectorEnv
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.training_runner import TrainingRunner
from maze.train.trainers.impala.impala_trainer import MultiStepIMPALA
from maze.utils.bcolors import BColors


@dataclasses.dataclass
class ImpalaRunner(TrainingRunner):
    """Common superclass for IMPALA runners, implementing the main training controls."""

    @override(TrainingRunner)
    def setup(self, cfg: DictConfig) -> None:
        """
        See :py:meth:`~maze.train.trainers.common.training_runner.TrainingRunner.setup`.
        """

        super().setup(cfg)

        # initialize actor critic model
        model = TorchActorCritic(
            policy=TorchPolicy(networks=self._model_composer.policy.networks,
                               distribution_mapper=self._model_composer.distribution_mapper,
                               device=cfg.algorithm.device),
            critic=self._model_composer.critic,
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
        eval_env_instance_seeds = [self.maze_seeding.generate_env_instance_seed() for _ in
                                   range(cfg.algorithm.eval_concurrency)]
        eval_env.seed(eval_env_instance_seeds)

        train_env_instance_seeds = [self.maze_seeding.generate_env_instance_seed() for _ in
                                    range(cfg.algorithm.num_actors)]
        train_agent_instance_seeds = [self.maze_seeding.generate_agent_instance_seed() for _ in
                                      range(cfg.algorithm.num_actors)]
        rollout_actors = self.create_distributed_rollout_actors(self.env_factory, rollout_policy,
                                                                cfg.algorithm.n_rollout_steps,
                                                                cfg.algorithm.num_actors,
                                                                cfg.algorithm.actors_batch_size,
                                                                cfg.algorithm.queue_out_of_sync_factor,
                                                                train_env_instance_seeds,
                                                                train_agent_instance_seeds)

        # initialize optimizer
        self._trainer = MultiStepIMPALA(
            model=model, rollout_actors=rollout_actors, eval_env=eval_env, options=cfg.algorithm
        )

        # initialize model from input_dir
        self._init_trainer_from_input_dir(trainer=self._trainer, state_dict_dump_file=self.state_dict_dump_file,
                                          input_dir=cfg.input_dir)

        # initialize best model selection
        self._model_selection = BestModelSelection(dump_file=self.state_dict_dump_file, model=model)

    @override(TrainingRunner)
    def run(
        self,
        n_epochs: Optional[int] = None,
        epoch_length: Optional[int] = None,
        deterministic_eval: Optional[bool] = None,
        eval_repeats: Optional[int] = None,
        patience: Optional[int] = None,
        model_selection: Optional[BestModelSelection] = None
    ) -> None:
        """
        See :py:meth:`~maze.train.trainers.common.training_runner.TrainingRunner.run`.
        :param n_epochs: number of epochs to train.
        :param epoch_length: number of updates per epoch.
        :param deterministic_eval: run evaluation in deterministic mode (argmax-policy)
        :param eval_repeats: number of evaluation trials
        :param patience: number of steps used for early stopping
        :param model_selection: Optional model selection class, receives model evaluation results
        """

        # train agent
        self._trainer.train(
            n_epochs=self._cfg.algorithm.n_epochs if n_epochs is None else n_epochs,
            epoch_length=self._cfg.algorithm.epoch_length if epoch_length is None else epoch_length,
            deterministic_eval=(
                self._cfg.algorithm.deterministic_eval if deterministic_eval is None else deterministic_eval
            ),
            eval_repeats=self._cfg.algorithm.eval_repeats if eval_repeats is None else eval_repeats,
            patience=self._cfg.algorithm.patience if patience is None else patience,
            model_selection=self._model_selection if model_selection is None else model_selection
        )

    @abstractmethod
    def create_distributed_eval_env(self,
                                    env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> VectorEnv:
        """The individual runners implement the setup of the distributed eval env"""

    @abstractmethod
    def create_distributed_rollout_actors(
            self,
            env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
            policy: TorchPolicy,
            n_rollout_steps: int,
            n_actors: int,
            batch_size: int,
            queue_out_of_sync_factor: float,
            env_instance_seeds: List[int],
            agent_instance_seeds: List[int]) -> DistributedActors:
        """The individual runners implement the setup of the distributed training rollout actors"""


@dataclasses.dataclass
class ImpalaDevRunner(ImpalaRunner):
    """Runner for single-threaded training, based on SequentialVectorEnv."""

    def create_distributed_rollout_actors(
            self,
            env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
            policy: TorchPolicy,
            n_rollout_steps: int,
            n_actors: int,
            batch_size: int,
            queue_out_of_sync_factor: float,
            env_instance_seeds: List[int],
            agent_instance_seeds: List[int]) -> SequentialDistributedActors:
        """Create dummy (sequentially-executed) actors."""
        return SequentialDistributedActors(env_factory, policy, n_rollout_steps, n_actors, batch_size,
                                           env_instance_seeds)

    def create_distributed_eval_env(self,
                                    env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> SequentialVectorEnv:
        """create single-threaded env distribution"""
        # fallback to a fixed number of pseudo-concurrent environments to avoid making this sequential execution
        # unnecessary slow on machines with a higher core number
        return SequentialVectorEnv([env_factory for _ in range(eval_concurrency)],
                                   logging_prefix=logging_prefix)


@dataclasses.dataclass
class ImpalaLocalRunner(ImpalaRunner):
    """Runner for locally distributed training, based on SubprocVectorEnv."""

    start_method: str
    """Type of start method used for multiprocessing ('forkserver', 'spawn', 'fork')."""

    def create_distributed_rollout_actors(
            self,
            env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
            policy: TorchPolicy,
            n_rollout_steps: int,
            n_actors: int,
            batch_size: int,
            queue_out_of_sync_factor: float,
            env_instance_seeds: List[int],
            agent_instance_seeds: List[int]) -> SubprocDistributedActors:
        """Create dummy (sequentially-executed) actors."""
        BColors.print_colored('Determinism by seeding of the IMPALA algorithm with the Local runner can not be '
                              'guarantied due to the asynchronicity of the implementation.', BColors.WARNING)
        return SubprocDistributedActors(env_factory, policy, n_rollout_steps, n_actors, batch_size,
                                        queue_out_of_sync_factor, self.start_method, env_instance_seeds,
                                        agent_instance_seeds)

    def create_distributed_eval_env(self,
                                    env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin]],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> SubprocVectorEnv:
        """create multi-process env distribution"""
        return SubprocVectorEnv([env_factory for _ in range(eval_concurrency)],
                                logging_prefix=logging_prefix)
