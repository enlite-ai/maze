"""Evaluation rollouts in the supplied env."""

from typing import Optional, Union

import numpy as np

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats import LogStatsLevel
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.train.trainers.common.evaluators.evaluator import Evaluator
from maze.train.parallelization.distributed_env.distributed_env import BaseDistributedEnv
from maze.train.trainers.common.model_selection.model_selection_base import ModelSelectionBase
from maze.train.utils.train_utils import unstack_numpy_list_dict


class RolloutEvaluator(Evaluator):
    """Evaluates a given policy by rolling it out and collecting the mean reward.

    :param eval_env: Distributed environment to run evaluation rollouts in.
    :param: n_episodes: Number of evaluation episodes to run. Note that the actual number might be slightly larger
                        due to the distributed nature of the environment.
    :param model_selection: Model selection to notify about the recorded rewards.
    """

    def __init__(self,
                 eval_env: Union[BaseDistributedEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv],
                 n_episodes: int,
                 model_selection: Optional[ModelSelectionBase]):
        self.eval_env = eval_env
        self.n_episodes = n_episodes
        self.model_selection = model_selection

    @override(Evaluator)
    def evaluate(self, policy: TorchPolicy) -> None:
        """Evaluate given policy (results are stored in stat logs) and dump the model if the reward improved.

        :param policy: Policy to evaluate
        """
        policy.eval()

        n_done_episodes = 0
        observations = self.eval_env.reset()

        while n_done_episodes < self.n_episodes:
            # Get the actor. Only synchronous envs are supported, i.e. the actor ID should always be the same
            # (regardless of the sub-step)
            actor_ids = set(self.eval_env.actor_id())
            assert len(actor_ids) == 1, "only synchronous environments are supported"

            # Sample action and take the step
            sampled_action = policy.compute_action(observations, policy_id=actor_ids.pop()[0], maze_state=None)
            observations, rewards, dones, infos = self.eval_env.step(unstack_numpy_list_dict(sampled_action))

            # Count done episodes
            n_done_episodes += np.count_nonzero(dones)

        # Enforce the epoch stats calculation (without calling increment_log_step() -- this is up to the trainer)
        self.eval_env.write_epoch_stats()

        # Notify the model selection if available
        if self.model_selection:
            reward = self.eval_env.get_stats_value(BaseEnvEvents.reward, LogStatsLevel.EPOCH, name="mean")
            self.model_selection.update(reward)
