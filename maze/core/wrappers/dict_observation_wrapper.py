"""Contains a dictionary observation space wrapper."""

from __future__ import annotations

from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.wrappers.wrapper import ObservationWrapper

import gymnasium as gym
import numpy as np


class DictObservationWrapper(ObservationWrapper[MazeEnv]):
    """Wraps a single observation into a dictionary space."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({'observation': env.observation_space})

    def observation(self, observation: np.ndarray | None) -> dict[str, np.ndarray] | None:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ObservationWrapper` interface."""
        # None indicates the absence of an observation; pass through without processing
        if observation is None:
            return None
        return {'observation': observation.astype(np.float32)}

    @override(SimulatedEnvMixin)
    def clone_from(self, env: DictObservationWrapper) -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
