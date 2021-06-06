"""Training code for OpenAI Evolution Strategies, based on https://github.com/openai/evolution-strategies-starter """
import itertools
import logging
import time
from typing import Optional, Iterable, Generator, Tuple, Union, Dict

import numpy as np
import torch
from typing.io import BinaryIO

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.log_stats.log_stats import LogStatsAggregator, LogStatsLevel, get_stats_logger, increment_log_step
from maze.core.utils.factory import Factory
from maze.train.trainers.common.model_selection.model_selection_base import ModelSelectionBase
from maze.train.trainers.common.trainer import Trainer
from maze.train.trainers.es.distributed.es_distributed_rollouts import ESDistributedRollouts, ESRolloutResult
from maze.train.trainers.es.es_algorithm_config import ESAlgorithmConfig
from maze.train.trainers.es.es_events import ESEvents
from maze.train.trainers.es.es_shared_noise_table import SharedNoiseTable
from maze.train.trainers.es.es_utils import get_flat_parameters
from maze.train.trainers.es.optimizers.base_optimizer import Optimizer

logger = logging.getLogger(__name__)


class ESTrainer(Trainer):
    """Trainer class for OpenAI Evolution Strategies.

    :param algorithm_config: Algorithm parameters.
    :param policy: Multi-step policy encapsulating the policy networks
    :param shared_noise: The noise table, with the same content for every worker and the master.
    :param normalization_stats: Normalization statistics as calculated by the NormalizeObservationWrapper.
    """

    def __init__(self,
                 algorithm_config: ESAlgorithmConfig,
                 policy: TorchPolicy,
                 shared_noise: SharedNoiseTable,
                 normalization_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]) -> None:
        super().__init__(algorithm_config)

        # --- training setup ---
        self.model_selection: Optional[ModelSelectionBase] = None
        self.policy = policy
        self.shared_noise = shared_noise
        self.normalization_stats = normalization_stats

        # setup the optimizer, now that the policy is available
        self.optimizer = Factory(Optimizer).instantiate(algorithm_config.optimizer)
        self.optimizer.setup(self.policy)

        # prepare statistics collection
        self.eval_stats = LogStatsAggregator(LogStatsLevel.EPOCH, get_stats_logger("eval"))
        self.train_stats = LogStatsAggregator(LogStatsLevel.EPOCH, get_stats_logger("train"))
        # injection of ES-specific events
        self.es_events = self.train_stats.create_event_topic(ESEvents)

    @override(Trainer)
    def train(
        self,
        distributed_rollouts: ESDistributedRollouts,
        n_epochs: Optional[int] = None,
        model_selection: Optional[ModelSelectionBase] = None
    ) -> None:
        """
        Run the ES training loop.
        :param distributed_rollouts: The distribution interface for experience collection.
        :param n_epochs: Number of epochs to train.
        :param model_selection: Optional model selection class, receives model evaluation results.
        """

        n_epochs = self.algorithm_config.n_epochs if n_epochs is None else n_epochs
        self.model_selection = model_selection

        for epoch in itertools.count():
            # check if we reached the max number of epochs
            if n_epochs and epoch == n_epochs:
                break

            print('********** Iteration {} **********'.format(epoch))

            step_start_time = time.time()

            # do the actual update step (disable autograd, as we calculate the gradient from the rollout returns)
            with torch.no_grad():
                self._update(distributed_rollouts)

            step_end_time = time.time()

            # log the step duration
            self.es_events.real_time(step_end_time - step_start_time)

            # update the epoch count
            increment_log_step()

    def load_state_dict(self, state_dict: Dict) -> None:
        """Set the model and optimizer state.
        :param state_dict: The state dict.
        """
        self.policy.load_state_dict(state_dict)

    @override(Trainer)
    def load_state(self, file_path: Union[str, BinaryIO]) -> None:
        """implementation of :class:`~maze.train.trainers.common.trainer.Trainer`
        """
        state_dict = torch.load(file_path, map_location=torch.device(self.policy.device))
        self.load_state_dict(state_dict)

    def _update(self, distributed_rollouts: ESDistributedRollouts):
        # Pop off results for the current task
        n_train_episodes, n_timesteps_popped = 0, 0

        # aggregate all collected training rollouts for this episode
        epoch_results = ESRolloutResult(is_eval=False)

        # obtain a generator from the distribution interface
        rollouts_generator = distributed_rollouts.generate_rollouts(
            policy=self.policy,
            max_steps=self.algorithm_config.max_steps,
            noise_stddev=self.algorithm_config.noise_stddev,
            normalization_stats=self.normalization_stats)

        # collect eval and training rollouts
        for result in rollouts_generator:
            if result.is_eval:
                # This was an eval job
                for e in result.episode_stats:
                    self.eval_stats.receive(e)
                continue

            # we received training experience from perturbed policy networks
            epoch_results.noise_indices.extend(result.noise_indices)
            epoch_results.episode_stats.extend(result.episode_stats)

            # update the training statistics
            for e in result.episode_stats:
                self.train_stats.receive(e)

                n_train_episodes += 1
                n_timesteps_popped += e[(BaseEnvEvents.reward, "count", None)]

            # continue until we collected enough episodes and timesteps
            if (n_train_episodes >= self.algorithm_config.n_rollouts_per_update and
                    n_timesteps_popped >= self.algorithm_config.n_timesteps_per_update):
                break

        # notify the model selection of the evaluation results
        eval_stats = self.eval_stats.reduce()
        if self.model_selection and len(eval_stats):
            reward = eval_stats[(BaseEnvEvents.reward, "mean", None)]
            self.model_selection.update(reward)

        # prepare returns, reshape the positive/negative antithetic estimation as (rollouts, 2)
        returns_n2 = np.array(
            [e[(BaseEnvEvents.reward, "sum", None)] for e in epoch_results.episode_stats]
        ).reshape(-1, 2)

        # improve robustness: weight by rank, not by reward
        proc_returns_n2 = self._compute_centered_ranks(returns_n2)

        # compute the gradient
        g = self._batched_weighted_sum(
            proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
            (self.shared_noise.get(idx, self.policy.num_params) for idx in epoch_results.noise_indices),
            batch_size=500
        )

        g /= n_train_episodes / 2.0

        # apply the weight update
        theta = get_flat_parameters(self.policy)
        update_ratio = self.optimizer.update(-g + self.algorithm_config.l2_penalty * theta.numpy())

        # statistics logging
        self.es_events.update_ratio(update_ratio)

        for i in self.policy.networks.keys():
            self.es_events.policy_grad_norm(policy_id=i, value=np.square(g).sum() ** 0.5)
            self.es_events.policy_norm(policy_id=i, value=np.square(theta).sum() ** 0.5)

    @classmethod
    def _iter_groups(cls, items: Iterable, group_size: int) -> Generator[Tuple, None, None]:
        assert group_size >= 1
        group = []
        for x in items:
            group.append(x)
            if len(group) == group_size:
                yield tuple(group)
                del group[:]
        if group:
            yield tuple(group)

    @classmethod
    def _batched_weighted_sum(cls,
                              weights: Iterable[float],
                              vectors: Iterable[np.ndarray],
                              batch_size: int) -> np.ndarray:
        """calculate a weighted sum of the given vectors, in steps of at most `batch_size` vectors"""
        # start with float, at the first operation numpy broadcasting takes care of the correct shape
        total: Union[np.array, float] = 0.

        for batch_weights, batch_vectors in zip(cls._iter_groups(weights, batch_size),
                                                cls._iter_groups(vectors, batch_size)):
            assert len(batch_weights) == len(batch_vectors) <= batch_size
            total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vectors, dtype=np.float32))

        return total

    @classmethod
    def _compute_ranks(cls, x: np.ndarray) -> np.ndarray:
        """
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        """
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    @classmethod
    def _compute_centered_ranks(cls, x):
        y = cls._compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return y
