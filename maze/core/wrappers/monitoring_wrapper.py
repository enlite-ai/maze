"""Contains a MazeEnv monitoring wrapper."""
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
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.time_env_mixin import TimeEnvMixin
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
        self.observation_events = self.core_env.context.event_service.create_event_topic(ObservationEvents)
        self.action_events = self.core_env.context.event_service.create_event_topic(ActionEvents)
        self.reward_events = self.core_env.context.event_service.create_event_topic(RewardEvents)

        # maintain for multi-step environments
        self._action_space: Optional[spaces.Dict] = None
        self._last_env_time: Optional[int] = None

    @override(BaseEnv)
    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, Dict[Any, Any]]:
        """Triggers logging events for observations, actions and reward.
        """

        # get identifier of current sub step
        substep_id, _ = self.env.actor_id() if isinstance(self.env, StructuredEnv) else (None, None)
        substep_name = f"step_key_{substep_id}" if substep_id is not None else None

        # take wrapped env step
        obs, rew, done, info = self.env.step(action)

        # OBSERVATION LOGGING
        # -------------------
        if self.observation_logging:
            # log processed observations
            for observation_name, observation_value in obs.items():
                self.observation_events.observation_processed(
                    step_key=substep_name, name=observation_name, value=observation_value)

            # log original observations
            for observation_name, observation_value in self.observation_original.items():
                self.observation_events.observation_original(
                    step_key=substep_name, name=observation_name, value=observation_value)

        # ACTION LOGGING
        # --------------
        if self.action_logging:
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

        # REWARD LOGGING
        # --------------
        if self.reward_logging:
            if not isinstance(self.env, TimeEnvMixin) or self.env.get_env_time() != self._last_env_time:
                self._last_env_time = self.env.get_env_time() if isinstance(self.env, TimeEnvMixin) \
                    else self._last_env_time + 1

                # record original reward only after actual maze env step
                if isinstance(self.env, MazeEnv):
                    self.reward_events.reward_original(value=self.env.reward_original)

        # update action space
        self._action_space = self.env.action_space

        return obs, rew, done, info

    def reset(self) -> ObservationType:
        """Resets the wrapper and returns the initial observation.

        :return: the initial observation after resetting.
        """
        # preserve action space
        self._action_space = self.env.action_space
        # reset env
        obs = self.env.reset()
        # update env time
        self._last_env_time = self.env.get_env_time() if isinstance(self.env, TimeEnvMixin) else 0
        return obs

    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Keep both actions and observation the same."""
        return self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)
