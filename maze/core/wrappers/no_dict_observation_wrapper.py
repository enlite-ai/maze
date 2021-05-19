"""Contains a dictionary-space removal observation wrapper."""
from typing import Dict, Union, Any, Tuple

import gym
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import ObservationWrapper, EnvType


class NoDictObservationWrapper(ObservationWrapper[Union[EnvType, StructuredEnvSpacesMixin]]):
    """Wraps observations by replacing the dictionary observation space with the sole contained sub-space.
    This wrapper is for example required when working with external frameworks not supporting dictionary observation
    spaces.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert len(env.observation_space.spaces) == 1
        self.observation_key = list(env.observation_space.spaces.keys())[0]

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_space(self) -> gym.spaces.Space:
        """The currently active gym observation space.
        """
        return self.env.observation_space.spaces[self.observation_key]

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """A dictionary of gym observation spaces, with policy IDs as keys.
        """
        return {k: v.spaces[self.observation_key] for k, v in self.env.observation_spaces_dict.items()}

    @override(ObservationWrapper)
    def observation(self, observation: Any) -> Any:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ObservationWrapper` interface.
        """
        return observation[self.observation_key]

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'NoDictObservationWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
