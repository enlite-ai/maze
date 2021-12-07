import multiprocessing
import signal

import cloudpickle
import torch

from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.time_limit_wrapper import TimeLimitWrapper
from maze.train.parallelization.broadcasting_container import BroadcastingContainer
from maze.train.trainers.es.distributed.es_rollout_wrapper import ESRolloutWorkerWrapper, ESAbortException
from maze.train.trainers.es.es_shared_noise_table import SharedNoiseTable


class ESSubprocWorker:
    def __init__(self,
                 pickled_env_factory: bytes,
                 pickled_policy: bytes,
                 shared_noise: SharedNoiseTable,
                 output_queue: multiprocessing.Queue,
                 broadcasting_container: BroadcastingContainer,
                 env_seed: int,
                 agent_seed: int,
                 is_eval_worker: bool
                 ):
        self.policy = cloudpickle.loads(pickled_policy)
        self.policy_version_counter = -1
        self.aux_data = None

        self.output_queue = output_queue
        self.broadcasting_container = broadcasting_container
        self.is_eval_worker = is_eval_worker

        env_factory = cloudpickle.loads(pickled_env_factory)
        self.env = env_factory()
        self.env = TimeLimitWrapper.wrap(self.env)
        if not isinstance(self.env, LogStatsWrapper):
            self.env = LogStatsWrapper.wrap(self.env)

        self.env.seed(env_seed)
        self.env = ESRolloutWorkerWrapper.wrap(env=self.env, shared_noise=shared_noise, agent_instance_seed=agent_seed)

    def run(self):
        while not self.broadcasting_container.stop_flag():
            self._update_policy_if_available()

            # limit the step count according to the task specification
            self.env.set_max_episode_steps(self.aux_data["max_steps"])
            if self.aux_data["normalization_stats"] is not None:
                self.env.set_normalization_statistics(self.aux_data["normalization_stats"])

            try:
                signal.signal(signal.SIGUSR1, self._abort_handler)
                with torch.no_grad():
                    if self.is_eval_worker:
                        result = self.env.generate_evaluation(self.policy)
                    else:
                        result = self.env.generate_training(self.policy, noise_stddev=self.aux_data["noise_stddev"])

                # Ignore abort during communication
                signal.signal(signal.SIGUSR1, signal.SIG_IGN)
                self.output_queue.put((self.policy_version_counter, result))
            except ESAbortException:
                if self.env.abort:
                    self.env.clear_abort()
                else:
                    raise

    def _update_policy_if_available(self) -> None:
        shared_policy_version_counter = self.broadcasting_container.policy_version()
        if self.policy_version_counter < shared_policy_version_counter:
            self.policy_version_counter = shared_policy_version_counter
            self.policy.load_state_dict(self.broadcasting_container.policy_state_dict())
            self.aux_data = self.broadcasting_container.aux_data()

    def _abort_handler(self, _signum, _frame):
        """Invoked if SIGUSR is send, interrupts current computation"""
        # Stop further signal handling
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        self.env.set_abort()
