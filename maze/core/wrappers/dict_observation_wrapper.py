"""Contains a dictionary observation space wrapper."""
from typing import Tuple, Any, Dict

import gym
import numpy as np
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin

from maze.core.wrappers.wrapper import ObservationWrapper


class DictObservationWrapper(ObservationWrapper[gym.Env]):
    """Wraps a single observation into a dictionary space.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({"observation": env.observation_space})

    def observation(self, observation: np.ndarray):
        """Implementation of :class:`~maze.core.wrappers.wrapper.ObservationWrapper` interface.
        """
        return {"observation": observation.astype(np.float32)}

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'DictObservationWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
