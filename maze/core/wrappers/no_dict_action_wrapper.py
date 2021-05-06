"""Contains a dictionary-space removal action wrapper."""
from typing import Dict, Union

import gym
import numpy as np
from maze.core.annotations import override
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import ActionWrapper, EnvType


class NoDictActionWrapper(ActionWrapper[Union[EnvType, StructuredEnvSpacesMixin]]):
    """Wraps actions by replacing the dictionary action space with the sole contained sub-space.
    This wrapper is for example required when working with external frameworks not supporting dictionary action spaces.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Dict)
        assert len(env.action_space.spaces) == 1
        self.action_key = list(env.action_space.spaces.keys())[0]

    @property
    @override(StructuredEnvSpacesMixin)
    def action_space(self) -> gym.spaces.Space:
        """The currently active gym action space.
        """
        return self.env.action_space.spaces[self.action_key]

    @property
    @override(StructuredEnvSpacesMixin)
    def action_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """A dictionary of gym action spaces, with policy IDs as keys.
        """
        return {k: v.spaces[self.action_key] for k, v in self.env.action_spaces_dict.items()}

    @override(ActionWrapper)
    def action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ActionWrapper` interface.
        """
        return {self.action_key: action}

    @override(ActionWrapper)
    def reverse_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ActionWrapper` interface.
        """
        return action[self.action_key]

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'NoDictActionWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
