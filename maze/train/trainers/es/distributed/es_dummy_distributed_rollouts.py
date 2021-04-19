"""Simplest possible implementation of the distributed rollout interface."""
from typing import Generator, Optional

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.observation_normalization.normalization_strategies.base import StructuredStatisticsType
from maze.core.wrappers.time_limit_wrapper import TimeLimitWrapper
from maze.train.trainers.es.distributed.es_distributed_rollouts import ESDistributedRollouts, ESRolloutResult
from maze.train.trainers.es.distributed.es_rollout_wrapper import ESRolloutWorkerWrapper
from maze.train.trainers.es.es_shared_noise_table import SharedNoiseTable


class ESDummyDistributedRollouts(ESDistributedRollouts):
    """Implementation of the ES distribution by running the rollouts synchronously in the same process."""

    def __init__(self, env: StructuredEnv, n_eval_rollouts: int, shared_noise: SharedNoiseTable,
                 agent_instance_seed: int):
        env = TimeLimitWrapper.wrap(env)
        env = LogStatsWrapper.wrap(env)
        self.env = ESRolloutWorkerWrapper.wrap(env=env, shared_noise=shared_noise,
                                               agent_instance_seed=agent_instance_seed)

        self.n_eval_rollouts = n_eval_rollouts

    @override(ESDistributedRollouts)
    def generate_rollouts(self,
                          policy: TorchPolicy,
                          max_steps: Optional[int],
                          noise_stddev: float,
                          normalization_stats: StructuredStatisticsType
                          ) -> Generator[ESRolloutResult, None, None]:
        """First execute a fixed number of eval rollouts and then continue with producing training samples."""
        self.env.set_max_episode_steps(max_steps)
        if normalization_stats:
            self.env.set_normalization_statistics(normalization_stats)

        for _ in range(self.n_eval_rollouts):
            yield self.env.generate_evaluation(policy)

        while True:
            yield self.env.generate_training(policy, noise_stddev)
