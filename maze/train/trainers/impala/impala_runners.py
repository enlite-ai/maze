"""Runner implementations for multi-step IMPALA"""
import copy
import dataclasses
from abc import abstractmethod
from typing import Union, Callable, List

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.utils.factory import Factory
from maze.train.parallelization.distributed_actors.distributed_actors import DistributedActors
from maze.train.parallelization.distributed_actors.sequential_distributed_actors import SequentialDistributedActors
from maze.train.parallelization.distributed_actors.subproc_distributed_actors import SubprocDistributedActors
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.parallelization.vector_env.subproc_vector_env import SubprocVectorEnv
from maze.train.parallelization.vector_env.vector_env import VectorEnv
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.training_runner import TrainingRunner
from maze.train.trainers.impala.impala_trainer import IMPALA
from maze.utils.bcolors import BColors
from omegaconf import DictConfig

from maze.utils.process import query_cpu


@dataclasses.dataclass
class ImpalaRunner(TrainingRunner):
    """Common superclass for IMPALA runners, implementing the main training controls."""

    eval_concurrency: int
    """ Number of concurrent evaluation envs """

    def __post_init__(self):
        """
        Adjusts initial values where necessary.
        """

        if self.eval_concurrency <= 0:
            self.eval_concurrency = query_cpu()

    @override(TrainingRunner)
    def setup(self, cfg: DictConfig) -> None:
        """
        See :py:meth:`~maze.train.trainers.common.training_runner.TrainingRunner.setup`.
        """

        super().setup(cfg)

        # initialize actor critic model
        model = TorchActorCritic(
            policy=self._model_composer.policy,
            critic=self._model_composer.critic,
            device=cfg.algorithm.device)

        # initialize best model selection
        self._model_selection = BestModelSelection(dump_file=self.state_dict_dump_file, model=model,
                                                   dump_interval=self.dump_interval)

        # policy for distributed actor rollouts
        #  - does not need critic
        #  - distributed actors are supported on CPU only
        #  - during training, new policy versions will be distributed to actors (as the training policy is updated)
        rollout_policy = copy.deepcopy(model.policy)
        rollout_policy.to("cpu")

        # initialize the env and enable statistics collection
        evaluator = None
        if cfg.algorithm.rollout_evaluator.n_episodes > 0:
            eval_env = self.create_distributed_eval_env(self.env_factory,
                                                        self.eval_concurrency,
                                                        logging_prefix="eval")
            eval_env_instance_seeds = [self.maze_seeding.generate_env_instance_seed() for _ in
                                       range(self.eval_concurrency)]
            eval_env.seed(eval_env_instance_seeds)

            # initialize rollout evaluator
            evaluator = Factory(base_type=RolloutEvaluator).instantiate(cfg.algorithm.rollout_evaluator,
                                                                        eval_env=eval_env,
                                                                        model_selection=self._model_selection)

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
        self._trainer = IMPALA(
            algorithm_config=cfg.algorithm,
            rollout_generator=rollout_actors,
            evaluator=evaluator,
            model=model,
            model_selection=self._model_selection,
        )

        # initialize model from input_dir
        self._init_trainer_from_input_dir(trainer=self._trainer, state_dict_dump_file=self.state_dict_dump_file,
                                          input_dir=cfg.input_dir)

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
                                    env_factory: Callable[[], Union[StructuredEnv, MazeEnv]],
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
                                    env_factory: Callable[[], Union[StructuredEnv, MazeEnv]],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> SubprocVectorEnv:
        """create multi-process env distribution"""
        return SubprocVectorEnv([env_factory for _ in range(eval_concurrency)],
                                logging_prefix=logging_prefix)
