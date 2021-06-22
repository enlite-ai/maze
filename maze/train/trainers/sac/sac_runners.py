"""Runner implementations for multi-step SAC"""
import copy
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, List

from maze.core.agent.policy import Policy
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_action_critic import TorchStateActionCritic
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.utils.factory import Factory
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.parallelization.vector_env.structured_vector_env import StructuredVectorEnv
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.training_runner import TrainingRunner
from maze.train.parallelization.distributed_actors.base_distributed_workers_with_buffer import \
    BaseDistributedWorkersWithBuffer
from maze.train.parallelization.distributed_actors.dummy_distributed_workers_with_buffer import \
    DummyDistributedWorkersWithBuffer
from maze.train.trainers.sac.sac_trainer import SAC
from omegaconf import DictConfig


@dataclass
class SACRunner(TrainingRunner):
    """
    Common superclass for SAC runners, implementing the main training controls.
    """

    eval_concurrency: int
    """ Number of concurrent evaluation envs """

    @override(TrainingRunner)
    def setup(self, cfg: DictConfig) -> None:
        """
        See :py:meth:`~maze.train.trainers.common.training_runner.TrainingRunner.setup`.
        """

        super().setup(cfg)

        assert isinstance(self._model_composer.critic, TorchStateActionCritic), \
            'Please specify a state action critic for SAC.'

        worker_policy = self._model_composer.policy
        worker_policy.to('cpu')

        model = TorchActorCritic(policy=copy.deepcopy(self._model_composer.policy),
                                 critic=self._model_composer.critic,
                                 device=cfg.algorithm.device)

        worker_env_instance_seeds = [self.maze_seeding.generate_env_instance_seed() for _ in
                                     range(cfg.algorithm.num_actors)]
        replay_buffer_seed = self.maze_seeding.generate_env_instance_seed()
        distributed_actors = self.create_distributed_rollout_workers(
            env_factory=self.env_factory, worker_policy=worker_policy, n_rollout_steps=cfg.algorithm.n_rollout_steps,
            n_workers=cfg.algorithm.num_actors, batch_size=cfg.algorithm.batch_size,
            replay_buffer_size=cfg.algorithm.replay_buffer_size, initial_buffer_size=cfg.algorithm.initial_buffer_size,
            initial_sampling_policy=cfg.algorithm.initial_sampling_policy,
            rollouts_per_iteration=cfg.algorithm.rollouts_per_iteration,
            split_rollouts_into_transitions=cfg.algorithm.split_rollouts_into_transitions,
            env_instance_seeds=worker_env_instance_seeds, replay_buffer_seed=replay_buffer_seed)

        # initialize best model selection
        self._model_selection = BestModelSelection(dump_file=self.state_dict_dump_file, model=model,
                                                   dump_interval=self.dump_interval)

        # initialize the env and enable statistics collection
        evaluator = None
        if cfg.algorithm.rollout_evaluator.n_episodes > 0:
            eval_env = self.create_distributed_eval_env(self.env_factory, self.eval_concurrency,
                                                        logging_prefix="eval")
            eval_env_instance_seeds = [self.maze_seeding.generate_env_instance_seed() for _ in
                                       range(self.eval_concurrency)]
            eval_env.seed(eval_env_instance_seeds)

            # initialize rollout evaluator
            evaluator = Factory(base_type=RolloutEvaluator).instantiate(cfg.algorithm.rollout_evaluator,
                                                                        eval_env=eval_env,
                                                                        model_selection=self._model_selection)

        # initialize trainer
        self._trainer = SAC(
            algorithm_config=cfg.algorithm,
            learner_model=model,
            distributed_actors=distributed_actors,
            model_selection=self._model_selection,
            evaluator=evaluator
        )

        # initialize model from input_dir
        self._init_trainer_from_input_dir(
            trainer=self._trainer, state_dict_dump_file=self.state_dict_dump_file, input_dir=cfg.input_dir
        )

    @abstractmethod
    def create_distributed_eval_env(self,
                                    env_factory: Callable[[], MazeEnv],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> StructuredVectorEnv:
        """The individual runners implement the setup of the distributed eval env

        :param env_factory: Factory function for envs to run rollouts on.
        :param eval_concurrency: The concurrency of the evaluation env.
        :param logging_prefix: The logging prefix to use for the envs.

        :return: A vector env.
        """

    @abstractmethod
    def create_distributed_rollout_workers(
            self, env_factory: Callable[[], MazeEnv],
            worker_policy: TorchPolicy, n_rollout_steps: int, n_workers: int, batch_size: int,
            replay_buffer_size: int, initial_buffer_size: int,
            initial_sampling_policy: Policy, rollouts_per_iteration: int,
            split_rollouts_into_transitions: bool,
            env_instance_seeds: List[int],
            replay_buffer_seed: int) -> BaseDistributedWorkersWithBuffer:
        """The individual runners implement the setup of the distributed training rollout actors.

        :param env_factory: Factory function for envs to run rollouts on.
        :param worker_policy: Structured policy to sample actions from.
        :param n_rollout_steps: Number of rollouts steps to record in one rollout.
        :param n_workers: Number of distributed workers to run simultaneously.
        :param batch_size: Size of the batch the rollouts are collected in.
        :param replay_buffer_size: The total size of the replay buffer.
        :param initial_buffer_size: The initial size of the replay buffer filled by sampling from the.
        :param initial_sampling_policy: Initial sampling policy to fill the buffer with :param initial_buffer_size
                                        initial samples.
        :param rollouts_per_iteration: The number of rollouts to collect each time the collect_rollouts method is
                                       called.
        :param split_rollouts_into_transitions: Specify whether to split rollouts into individual transitions.
        :param env_instance_seeds: The seed for each of the workers env.
        :param replay_buffer_seed: The seed for the replay buffer.

        :return: A BaseDistributedWorkersWithBuffer object.
        """


@dataclass
class SACDevRunner(SACRunner):
    """
    Runner for single-threaded training, based on SequentialVectorEnv.
    """

    @override(SACRunner)
    def create_distributed_rollout_workers(
            self, env_factory: Callable[[], MazeEnv],
            worker_policy: TorchPolicy, n_rollout_steps: int, n_workers: int, batch_size: int,
            replay_buffer_size: int, initial_buffer_size: int,
            initial_sampling_policy: Policy, rollouts_per_iteration: int,
            split_rollouts_into_transitions: bool,
            env_instance_seeds: List[int],
            replay_buffer_seed: int
    ) -> DummyDistributedWorkersWithBuffer:
        """Create dummy (sequentially-executed) actors."""
        return DummyDistributedWorkersWithBuffer(env_factory, worker_policy, n_rollout_steps, n_workers, batch_size,
                                                 rollouts_per_iteration, initial_sampling_policy,
                                                 replay_buffer_size, initial_buffer_size,
                                                 split_rollouts_into_transitions, env_instance_seeds,
                                                 replay_buffer_seed)

    @override(SACRunner)
    def create_distributed_eval_env(self,
                                    env_factory: Callable[[], MazeEnv],
                                    eval_concurrency: int,
                                    logging_prefix: str
                                    ) -> SequentialVectorEnv:
        """create single-threaded env distribution"""
        # fallback to a fixed number of pseudo-concurrent environments to avoid making this sequential execution
        # unnecessary slow on machines with a higher core number
        return SequentialVectorEnv([env_factory for _ in range(eval_concurrency)], logging_prefix=logging_prefix)
