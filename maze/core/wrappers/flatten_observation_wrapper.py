"""Contains a generic observation-space flattening wrapper."""
from typing import Dict, Union, Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import utils as space_utils

from maze.core.annotations import override
from maze.core.env.base_env import BaseEnv
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import ObservationWrapper, EnvType


class FlattenDictObservationWrapper(ObservationWrapper[Union[EnvType, StructuredEnvSpacesMixin]]):
    """
    Flattens arbitrary observation spaces (Dict, nested Dict, Tuple, Discrete, etc.)
    into a single 1-D Box using gymnasium's native flatten utilities.

    :param env: The environment to wrap
    """

    def __init__(self, env):
        super().__init__(env)

        self._obs_space = env.observation_space
        # flat space
        self._flat_space = space_utils.flatten_space(self._obs_space)

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_space(self) -> gym.spaces.Box:
        """
        Returns the flattened observation space as a gym.spaces.Box
        """
        return self._flat_space

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Box]:
        """
        A dictionary of gym observation spaces, with policy IDs as keys.
        """
        return {
            k: space_utils.flatten_space(dict_space)
            for k, dict_space in self.env.observation_spaces_dict.items()
        }

    @override(ObservationWrapper)
    def observation(self, observation: Any) -> np.ndarray:
        """
        Flattens the observation using gymnasium's flatten utilities.
        """
        return space_utils.flatten(self._obs_space, observation).astype(np.float32)

    @override(SimulatedEnvMixin)
    def clone_from(self, env: "FlattenDictObservationWrapper") -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)

    @override(BaseEnv)
    def reset(self) -> Any:
        """
        Override the reset method to flatten the observation.
        """
        observation = self.env.reset()
        return self.observation(observation)

    def step(self, action) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """
        Override the step method to flatten the observation.
        """
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info