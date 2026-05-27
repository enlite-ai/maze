"""Contains a generic observation-space flattening wrapper."""

from __future__ import annotations

from typing import Any

from maze.core.annotations import override
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import EnvType, ObservationWrapper

import gymnasium as gym
import numpy as np
from gymnasium.spaces import utils as space_utils


class FlattenDictObservationWrapper(ObservationWrapper[EnvType | StructuredEnvSpacesMixin]):
    """
    Flattens arbitrary observation spaces (Dict, nested Dict, Tuple, Discrete, etc.)
    into a single 1-D Box using gymnasium's native flatten utilities.

    :param env: The environment to wrap
    """

    def __init__(self, env):
        super().__init__(env)

        self._original_obs_space = env.observation_space
        # flat space
        self._flat_space = space_utils.flatten_space(self._original_obs_space)

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_space(self) -> gym.spaces.Box:
        """
        Returns the flattened observation space as a gym.spaces.Box

        :return: The flattened observation space
        """
        return self._flat_space

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_spaces_dict(self) -> dict[int | str, gym.spaces.Box]:
        """
        A dictionary of gym observation spaces, with policy IDs as keys.

        :return: A dictionary of gym observation spaces, with policy IDs as keys.
        """
        return {k: space_utils.flatten_space(dict_space) for k, dict_space in self.env.observation_spaces_dict.items()}

    @override(ObservationWrapper)
    def observation(self, observation: Any) -> np.ndarray | None:
        """
        Flattens the observation using gymasium flatten utilities.

        :param observation: The observation to be flattened
        :return: The flattened observation
        """
        # None indicates the absence of an observation; pass through without processing
        if observation is None:
            return None
        return space_utils.flatten(self._original_obs_space, observation).astype(np.float32)

    @override(SimulatedEnvMixin)
    def clone_from(self, env: FlattenDictObservationWrapper) -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
