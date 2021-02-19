"""Generate observation statistics for any gym environment."""
from typing import TypeVar, Union, Any, Tuple, Dict, Optional

import gym
from maze.core.annotations import override
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.structured_env import StructuredEnv
from maze.core.log_events.observation_events import ObservationEvents
from maze.core.wrappers.wrapper import Wrapper


class ObservationLoggingWrapper(Wrapper[BaseEnv]):
    """A observation logging wrapper for :class:`~maze.core.env.base_env.BaseEnv`.

    :param env: The environment to wrap.
    """

    T = TypeVar("T")

    def __init__(self, env: Union[gym.Env, BaseEnv]):
        """Avoid calling this constructor directly, use :method:`wrap` instead."""
        super().__init__(env)

        # create observation event topic
        self.observation_events = self.core_env.context.event_service.create_event_topic(ObservationEvents)

    @override(BaseEnv)
    def step(self, action: Any) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Create the observation logs on every step
        """
        substep_name, _ = self.env.actor_id() if isinstance(self.env, StructuredEnv) else None
        obs, rew, done, info = self.env.step(action)

        # prepare observation visualizations
        for observation_name, observation_value in obs.items():
            self.observation_events.observation_processed(
                step_key=substep_name, name=observation_name, value=observation_value)

        # prepare visualization of original observations
        for observation_name, observation_value in self.observation_original.items():
            self.observation_events.observation_original(
                step_key=substep_name, name=observation_name, value=observation_value)

        return obs, rew, done, info

    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType], maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool)\
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Keep both actions and observation the same."""
        return self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)