"""Evaluation rollouts in the supplied env."""
from typing import Optional

import numpy as np

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.base_env_events import BaseEnvEvents
from maze.core.log_stats.log_stats import LogStatsLevel
from maze.train.parallelization.vector_env.structured_vector_env import StructuredVectorEnv
from maze.train.trainers.common.evaluators.evaluator import Evaluator
from maze.train.trainers.common.model_selection.model_selection_base import ModelSelectionBase


class RolloutEvaluator(Evaluator):
    """Evaluates a given policy by rolling it out and collecting the mean reward.

    :param eval_env: Distributed environment to run evaluation rollouts in.
    :param n_episodes: Number of evaluation episodes to run. Note that the actual number might be slightly larger
                        due to the distributed nature of the environment.
    :param model_selection: Model selection to notify about the recorded rewards.
    """

    def __init__(self,
                 eval_env: StructuredVectorEnv,
                 n_episodes: int,
                 model_selection: Optional[ModelSelectionBase],
                 deterministic: bool = False):
        self.eval_env = eval_env
        self.n_episodes = n_episodes
        self.model_selection = model_selection
        self.deterministic = deterministic

    @override(Evaluator)
    def evaluate(self, policy: TorchPolicy) -> None:
        """Evaluate given policy (results are stored in stat logs) and dump the model if the reward improved.

        :param policy: Policy to evaluate
        """
        policy.eval()

        n_done_episodes = 0
        observations = self.eval_env.reset()

        # Clear epoch stats in case there were some unfinished episodes from the previous evaluation round
        self.eval_env.clear_epoch_stats()

        while n_done_episodes < self.n_episodes:
            # Sample action and take the step
            sampled_action = policy.compute_action(observations, actor_id=self.eval_env.actor_id(), maze_state=None,
                                                   deterministic=self.deterministic)
            observations, rewards, dones, infos = self.eval_env.step(sampled_action)

            # Count done episodes
            n_done_episodes += np.count_nonzero(dones)

        # Enforce the epoch stats calculation (without calling increment_log_step() -- this is up to the trainer)
        self.eval_env.write_epoch_stats()

        # Notify the model selection if available
        if self.model_selection:
            reward = self.eval_env.get_stats_value(BaseEnvEvents.reward, LogStatsLevel.EPOCH, name="mean")
            self.model_selection.update(reward)
