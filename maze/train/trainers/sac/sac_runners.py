"""Runner implementations for multi-step SAC"""
import copy
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Union

from omegaconf import DictConfig
from torch.utils.data import Dataset

from maze.core.agent.policy import Policy
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_action_critic import TorchStateActionCritic
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.log_stats.log_stats import LogStatsAggregator, LogStatsLevel, get_stats_logger
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.trajectory_recording.datasets.in_memory_dataset import InMemoryDataset
from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord
from maze.core.utils.config_utils import SwitchWorkingDirectoryToInput
from maze.core.utils.factory import Factory
from maze.train.parallelization.distributed_actors.base_distributed_workers_with_buffer import \
    BaseDistributedWorkersWithBuffer
from maze.train.parallelization.distributed_actors.dummy_distributed_workers_with_buffer import \
    DummyDistributedWorkersWithBuffer
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.parallelization.vector_env.structured_vector_env import StructuredVectorEnv
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.replay_buffer.replay_buffer import BaseReplayBuffer
from maze.train.trainers.common.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from maze.train.trainers.common.training_runner import TrainingRunner
from maze.train.trainers.sac.sac_trainer import SAC
from maze.utils.process import query_cpu


@dataclass
class SACRunner(TrainingRunner):
    """
    Common superclass for SAC runners, implementing the main training controls.
    """

    eval_concurrency: int
    """ Number of concurrent evaluation envs """

    initial_demonstration_trajectories: DictConfig
    """Optionally a trajectory, list of trajectories, a dir or list of directories can be given to fill the replay 
    buffer with. If this is not given (is None) the initial replay buffer is filled with the (algorithm) specified
    initial_sampling_policy"""

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

        # Init replay buffer
        replay_buffer = UniformReplayBuffer(cfg.algorithm.replay_buffer_size, seed=replay_buffer_seed)
        if cfg.runner.initial_demonstration_trajectories:
            self.load_replay_buffer(replay_buffer=replay_buffer, cfg=cfg)
        else:
            self.init_replay_buffer(
                replay_buffer=replay_buffer, initial_sampling_policy=cfg.algorithm.initial_sampling_policy,
                initial_buffer_size=cfg.algorithm.initial_buffer_size, replay_buffer_seed=replay_buffer_seed,
                split_rollouts_into_transitions=cfg.algorithm.split_rollouts_into_transitions,
                n_rollout_steps=cfg.algorithm.n_rollout_steps, env_factory=self.env_factory)

        distributed_actors = self.create_distributed_rollout_workers(
            env_factory=self.env_factory, worker_policy=worker_policy, n_rollout_steps=cfg.algorithm.n_rollout_steps,
            n_workers=cfg.algorithm.num_actors, batch_size=cfg.algorithm.batch_size,
            rollouts_per_iteration=cfg.algorithm.rollouts_per_iteration,
            split_rollouts_into_transitions=cfg.algorithm.split_rollouts_into_transitions,
            env_instance_seeds=worker_env_instance_seeds, replay_buffer=replay_buffer)

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

    def load_replay_buffer(self, replay_buffer: BaseReplayBuffer,
                           cfg: DictConfig) -> None:
        """Load the given trajectories as a dataset and fill the buffer with these trajectories.

        :param replay_buffer: The replay buffer to fill.
        :param cfg: The dict config of the experiment.
        """

        print(f'******* Starting to fill the replay buffer with trajectories from path: '
              f'{self.initial_demonstration_trajectories.input_data} *******')
        with SwitchWorkingDirectoryToInput(cfg.input_dir):
            dataset = Factory(base_type=Dataset).instantiate(self.initial_demonstration_trajectories,
                                                             conversion_env_factory=self.env_factory)
        assert isinstance(dataset, InMemoryDataset), 'Only in memory dataset supported at this point'

        if cfg.algorithm.split_rollouts_into_transitions:
            for step_record in dataset.step_records:
                assert step_record.next_observations is not None, "Next observations are required for sac"
                assert all(map(lambda x: x is not None, step_record.next_observations)), \
                    "Next observations are required for sac"
                replay_buffer.add_transition(step_record)
        else:
            for idx, trajectory_reference in enumerate(dataset.trajectory_references):
                traj = SpacesTrajectoryRecord(id=idx)
                traj.step_records = dataset.step_records[trajectory_reference]
                replay_buffer.add_transition(traj)

    @staticmethod
    def init_replay_buffer(replay_buffer: BaseReplayBuffer, initial_sampling_policy: Union[DictConfig, Policy],
                           initial_buffer_size: int, replay_buffer_seed: int,
                           split_rollouts_into_transitions: bool, n_rollout_steps: int,
                           env_factory: Callable[[], MazeEnv]) -> None:
        """Fill the buffer with initial_buffer_size rollouts by rolling out the initial_sampling_policy.

        :param replay_buffer: The replay buffer to use.
        :param initial_sampling_policy: The initial sampling policy used to fill the buffer to the initial fill state.
        :param initial_buffer_size: The initial size of the replay buffer filled by sampling from the initial sampling
            policy.
        :param replay_buffer_seed: A seed for initializing and sampling from the replay buffer.
        :param split_rollouts_into_transitions: Specify whether to split rollouts into individual transitions.
        :param n_rollout_steps: Number of rollouts steps to record in one rollout.
        :param env_factory: Factory function for envs to run rollouts on.
        """

        # Create the log stats aggregator for collecting kpis of initializing the replay buffer
        epoch_stats = LogStatsAggregator(LogStatsLevel.EPOCH)
        replay_stats_logger = get_stats_logger('init_replay_buffer')
        epoch_stats.register_consumer(replay_stats_logger)

        dummy_env = env_factory()
        dummy_env.seed(replay_buffer_seed)
        sampling_policy: Policy = \
            Factory(Policy).instantiate(initial_sampling_policy, action_spaces_dict=dummy_env.action_spaces_dict)
        sampling_policy.seed(replay_buffer_seed)
        rollout_generator = RolloutGenerator(env=dummy_env,
                                             record_next_observations=True,
                                             record_episode_stats=True)

        print(f'******* Starting to fill the replay buffer with {initial_buffer_size} transitions *******')
        while len(replay_buffer) < initial_buffer_size:
            trajectory = rollout_generator.rollout(policy=sampling_policy, n_steps=n_rollout_steps)

            if split_rollouts_into_transitions:
                replay_buffer.add_rollout(trajectory)
            else:
                replay_buffer.add_transition(trajectory)

            # collect episode statistics
            for step_record in trajectory.step_records:
                if step_record.episode_stats is not None:
                    epoch_stats.receive(step_record.episode_stats)

        # Print the kpis from initializing the replay buffer
        epoch_stats.reduce()
        # Remove the consumer again from the aggregator
        epoch_stats.remove_consumer(replay_stats_logger)

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
            rollouts_per_iteration: int,
            split_rollouts_into_transitions: bool,
            env_instance_seeds: List[int],
            replay_buffer: BaseReplayBuffer) -> BaseDistributedWorkersWithBuffer:
        """The individual runners implement the setup of the distributed training rollout actors.

        :param env_factory: Factory function for envs to run rollouts on.
        :param worker_policy: Structured policy to sample actions from.
        :param n_rollout_steps: Number of rollouts steps to record in one rollout.
        :param n_workers: Number of distributed workers to run simultaneously.
        :param batch_size: Size of the batch the rollouts are collected in.
        :param rollouts_per_iteration: The number of rollouts to collect each time the collect_rollouts method is
                                       called.
        :param split_rollouts_into_transitions: Specify whether to split rollouts into individual transitions.
        :param env_instance_seeds: The seed for each of the workers env.
        :param replay_buffer: The replay buffer to use.

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
            rollouts_per_iteration: int, split_rollouts_into_transitions: bool, env_instance_seeds: List[int],
            replay_buffer: BaseReplayBuffer
    ) -> DummyDistributedWorkersWithBuffer:
        """Create dummy (sequentially-executed) actors."""
        return DummyDistributedWorkersWithBuffer(env_factory=env_factory, worker_policy=worker_policy,
                                                 n_rollout_steps=n_rollout_steps, n_workers=n_workers,
                                                 batch_size=batch_size, rollouts_per_iteration=rollouts_per_iteration,
                                                 split_rollouts_into_transitions=split_rollouts_into_transitions,
                                                 env_instance_seeds=env_instance_seeds, replay_buffer=replay_buffer)

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
