"""Contains a reward scaling wrapper."""
import copy

import numpy as np

from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.utils.stats_utils import CumulativeMovingMeanStd
from maze.core.wrappers.wrapper import RewardWrapper


class ReturnNormalizationRewardWrapper(RewardWrapper[MazeEnv]):
    """Normalizes step reward by dividing through the standard deviation of the discounted return.

    Implementation adopted from: https://github.com/MadryLab/implementation-matters

    :param env: The underlying environment.
    :param gamma: The discounting factor (e.g., 0.99).
    :param epsilon: Ensures numerical stability and avoid division by zero (e.g., 1e-8).
    """

    def __init__(self, env: MazeEnv, gamma: float, epsilon: float):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon

        self._return = 0
        self._return_stats = CumulativeMovingMeanStd(epsilon=self.epsilon)

    @override(RewardWrapper)
    def reward(self, reward: float) -> float:
        """implementation of :class:`~maze.core.wrappers.wrapper.RewardWrapper`
        """

        # update
        self._return = self._return * self.gamma + reward
        self._return_stats.update(self._return)

        # normalize reward
        return float(reward / np.sqrt(self._return_stats.var + self.epsilon))

    def reset(self):
        """implementation of :class:`~maze.core.wrappers.wrapper.RewardWrapper`
        """
        self._return = 0
        return self.env.reset()

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'ReturnNormalizationRewardWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self._return = env._return
        self._return_stats = copy.deepcopy(env._return_stats)
        self.env.clone_from(env)
