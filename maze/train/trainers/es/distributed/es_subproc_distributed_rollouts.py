import logging
import multiprocessing
import os
import signal
from collections import Callable
from multiprocessing.context import BaseContext
from typing import Union, Optional, Generator

from maze.core.agent.policy import Policy
from maze.core.agent.torch_model import TorchModel
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.observation_normalization.normalization_strategies.base import StructuredStatisticsType
from maze.train.parallelization.broadcasting_container import BroadcastingManager, BroadcastingContainer
from maze.train.trainers.es.distributed.es_distributed_rollouts import ESDistributedRollouts, ESRolloutResult
from maze.train.trainers.es.distributed.es_subproc_worker import ESSubprocWorker
from maze.train.trainers.es.es_shared_noise_table import SharedNoiseTable


class ESSubprocDistributedRollouts(ESDistributedRollouts):
    def __init__(self,
                 env_factory: Callable[[], Union[MazeEnv]],
                 n_training_workers: int,
                 n_eval_workers: int,
                 shared_noise: SharedNoiseTable,
                 env_seed: int,
                 agent_seed: int,
                 start_method: str = None
                 ):
        ctx = self._get_multiprocessing_context(start_method)
        self.worker_output_queue = ctx.Queue()
        self.broadcasting_container = self._create_broadcasting_container()

        self.workers = []
        for worker_id in range(n_eval_workers + n_training_workers):
            process = ctx.Process(target=self._launch_worker, kwargs=dict(
                env_factory=env_factory,
                shared_noise=shared_noise,
                output_queue=self.worker_output_queue,
                broadcasting_container=self.broadcasting_container,
                env_seed=env_seed,
                agent_seed=agent_seed,
                is_eval_worker=worker_id < n_eval_workers
            ))
            self.workers.append(process)

        self._workers_started = False

    @override(ESDistributedRollouts)
    def generate_rollouts(self,
                          policy: Union[Policy, TorchModel],
                          max_steps: Optional[int],
                          noise_stddev: float,
                          normalization_stats: StructuredStatisticsType
                          ) -> Generator[ESRolloutResult, None, None]:
        """First execute a fixed number of eval rollouts and then continue with producing training samples."""
        self.broadcasting_container.set_aux_data(dict(
            normalizatin_stats=normalization_stats,
            max_steps=max_steps,
            noise_stddev=noise_stddev
        ))

        self.broadcasting_container.set_policy_state_dict(policy.state_dict())
        current_policy_version = self.broadcasting_container.policy_version()

        if not self._workers_started:
            # Start workers if not yet started
            for worker in self.workers:
                worker.start()
        else:
            # Notify workers to abort current rollout and start a new one
            for worker in self.workers:
                os.kill(worker.pid, signal.SIGUSR1)

        while True:
            policy_version, result = self.worker_output_queue.get()

            if current_policy_version == policy_version:
                yield result

    @staticmethod
    def _launch_worker(**kwargs) -> None:
        try:
            ESSubprocWorker(**kwargs).run()
        except Exception as e:
            logging.exception(e)

    @staticmethod
    def _get_multiprocessing_context(start_method: Optional[str]) -> BaseContext:
        if start_method is None:
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        return multiprocessing.get_context(start_method)

    @staticmethod
    def _create_broadcasting_container() -> BroadcastingContainer:
        BroadcastingManager.register('BroadcastingContainer', BroadcastingContainer)
        manager = BroadcastingManager()
        manager.start()
        return manager.BroadcastingContainer()
