"""Contains a dictionary-space removal action wrapper."""
from typing import Union, Dict, Any, Tuple, Optional

import gym
import numpy as np
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import EnvType, Wrapper


class NoDictSpacesWrapper(Wrapper[Union[EnvType, StructuredEnvSpacesMixin]]):
    """Wraps observations and actions by replacing dictionary spaces with the sole contained sub-space.
    This wrapper is for example required when working with external frameworks not supporting dictionary spaces.
    """

    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert len(env.observation_space.spaces) == 1
        self.observation_key = list(env.observation_space.spaces.keys())[0]

        assert isinstance(env.action_space, gym.spaces.Dict)
        assert len(env.action_space.spaces) == 1
        self.action_key = list(env.action_space.spaces.keys())[0]

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_space(self) -> gym.spaces.Space:
        """The currently active gym observation space.
        """
        return self.env.observation_space.spaces[self.observation_key]

    @property
    @override(StructuredEnvSpacesMixin)
    def action_space(self) -> gym.spaces.Space:
        """The currently active gym action space.
        """
        return self.env.action_space.spaces[self.action_key]

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """A dictionary of gym observation spaces, with policy IDs as keys.
        """
        return {k: v.spaces[self.observation_key] for k, v in self.env.observation_spaces_dict.items()}

    @property
    @override(StructuredEnvSpacesMixin)
    def action_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """A dictionary of gym action spaces, with policy IDs as keys.
        """
        return {k: v.spaces[self.action_key] for k, v in self.env.action_spaces_dict.items()}

    def observation(self, observation: Any) -> Any:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ObservationWrapper` interface.
        """
        return observation[self.observation_key]

    def action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ActionWrapper` interface.
        """
        return {self.action_key: action}

    def reverse_action(self, action: Dict[str, np.ndarray]) -> np.ndarray:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ActionWrapper` interface.
        """
        return action[self.action_key]

    @override(BaseEnv)
    def reset(self) -> Any:
        """Intercept ``BaseEnv.reset`` and map observation."""
        observation = self.env.reset()
        return self.observation(observation)

    def step(self, action) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Intercept ``BaseEnv.step`` and map observation."""
        observation, reward, done, info = self.env.step(self.action(action))
        return self.observation(observation), reward, done, info

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType], maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Convert the observations, reverse the actions."""

        obs_dict, act_dict = self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)

        if act_dict is not None:
            act_dict = {policy_id: self.reverse_action(action) for policy_id, action in act_dict.items()}

        if obs_dict is not None:
            obs_dict = {policy_id: self.observation(obs) for policy_id, obs in obs_dict.items()}

        return obs_dict, act_dict

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'NoDictSpacesWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
