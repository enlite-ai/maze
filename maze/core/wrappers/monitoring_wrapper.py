"""Contains a MazeEnv monitoring wrapper."""
import warnings
from typing import TypeVar, Union, Any, Tuple, Dict, Optional

import numpy as np
from gym import spaces

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env import StructuredEnv
from maze.core.log_events.monitoring_events import ActionEvents, RewardEvents, ObservationEvents
from maze.core.wrappers.wrapper import Wrapper


class MazeEnvMonitoringWrapper(Wrapper[MazeEnv]):
    """A MazeEnv monitoring wrapper logging events for observations, actions and rewards.

    :param env: The environment to wrap.
    :param observation_logging: If True observation events are logged.
    :param action_logging: If True action events are logged.
    :param reward_logging: If True additional reward events are logged.
    """

    T = TypeVar("T")

    def __init__(self, env: MazeEnv, observation_logging: bool, action_logging: bool, reward_logging: bool):
        """Avoid calling this constructor directly, use :method:`wrap` instead."""
        super().__init__(env)

        self.observation_logging = observation_logging
        self.action_logging = action_logging
        self.reward_logging = reward_logging

        # create event topics
        if self.observation_logging:
            self.observation_events = self.core_env.context.event_service.create_event_topic(ObservationEvents)
        if self.action_logging:
            self.action_events = self.core_env.context.event_service.create_event_topic(ActionEvents)
        if self.reward_logging:
            self.reward_events = self.core_env.context.event_service.create_event_topic(RewardEvents)

        # maintain for multi-step environments
        self._action_space: Optional[spaces.Dict] = None

    @override(BaseEnv)
    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, Dict[Any, Any]]:
        """Triggers logging events for observations, actions and reward.
        """

        substep_name = self._get_substep_name()

        # take wrapped env step
        obs, rew, done, info = self.env.step(action)

        if self.action_logging:
            self._log_action(substep_name, action)

        if self.observation_logging:
            self._log_observation(substep_name, obs)

        if self.reward_logging:
            self.reward_events.reward_processed(step_key=substep_name, value=rew)

        # update action space
        self._action_space = self.env.action_space

        return obs, rew, done, info

    @override(BaseEnv)
    def reset(self) -> ObservationType:
        """Resets the wrapper and returns the initial observation.

        :return: the initial observation after resetting.
        """
        # preserve action space
        self._action_space = self.env.action_space

        # reset env
        obs = self.env.reset()

        if self.observation_logging:
            substep_name = self._get_substep_name()
            self._log_observation(substep_name, obs)

        return obs

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Keep both actions and observation the same."""
        return self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)

    def _get_substep_name(self):
        if isinstance(self.env, StructuredEnv):
            return f"step_key_{self.env.actor_id()[0]}"
        else:
            return None

    def _log_observation(self, substep_name: Union[str, int], observation: ObservationType) -> None:

        # log processed observations
        for observation_name, observation_value in observation.items():
            self.observation_events.observation_processed(
                step_key=substep_name, name=observation_name, value=observation_value)

        # log original observations
        for observation_name, observation_value in self.observation_original.items():
            self.observation_events.observation_original(
                step_key=substep_name, name=observation_name, value=observation_value)

    def _log_action(self, substep_name: Union[str, int], action: ActionType) -> None:
        assert isinstance(action, Dict), "The action space of your env has to be a dict action space."

        for actor_name, actor_action_space in self._action_space.spaces.items():
            # If actor_name is not in action, cycle loop
            if actor_name not in action:
                continue
            # Check for discrete sub-action space
            if isinstance(actor_action_space, spaces.Discrete):
                # For some policies, the action is a np.ndarray, for others the action is an int. To support both cases,
                # transform the actions to int (done via v.item() below)
                actor_action = action[actor_name]
                if isinstance(actor_action, np.ndarray):
                    actor_action = actor_action.item()
                self.action_events.discrete_action(step_key=substep_name, name=actor_name,
                                                   value=actor_action, action_dim=actor_action_space.n)

            # Check for multi-discrete sub-action space
            elif isinstance(actor_action_space, spaces.MultiDiscrete):
                for sub_actor_idx, sub_action_space in enumerate(actor_action_space.nvec):
                    # a multi-discrete action space consists of several discrete action spaces
                    self.action_events.discrete_action(step_key=substep_name,
                                                       name=f'{actor_name}_{sub_actor_idx}',
                                                       value=action[actor_name][..., sub_actor_idx],
                                                       action_dim=actor_action_space.nvec[sub_actor_idx])

            elif isinstance(actor_action_space, spaces.MultiBinary):
                actor_action = action[actor_name]
                assert isinstance(actor_action, np.ndarray)
                self.action_events.multi_binary_action(step_key=substep_name, name=actor_name,
                                                       value=actor_action, num_binary_actions=actor_action_space.n)

            # Check for box sub-action space
            elif isinstance(actor_action_space, spaces.Box):
                actor_action = action[actor_name]
                self.action_events.continuous_action(step_key=substep_name, name=actor_name, value=actor_action)

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'MazeEnvMonitoringWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        warnings.warn("Try to avoid wrappers such as the 'MazeEnvMonitoringWrapper'"
                      "when working with simulated envs to reduce overhead.")
        self.env.clone_from(env)
