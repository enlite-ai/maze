"""
Includes the implementation of the dummy structured environment.
"""

from typing import Any, Dict, Union, Tuple, Callable, Optional

import gym
import numpy as np

from maze.core.env.maze_env import MazeEnv
from maze.core.env.structured_env import StructuredEnv, StepKeyType, ActorID
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import Wrapper


def filter_dict(el: Dict, callback: Callable[[str], bool]) -> Dict:
    """
    Filters dicts by key using a callback

    example:
    filter_dict({"a": 10, "aa": 11, "b": 20}, lambda key: key.startswith("a"))
    returns: {'a': 10, 'aa': 11}

    :param el: The dict you want to filter
    :param callback: The callback to filter the dict keys
    :return: The filtered dict
    """
    keys = filter(callback, el.keys())
    return {key: el[key] for key in keys}


def filter_dict_starts_with(el: Dict, start_with: str) -> Dict:
    """
    Filters a dict by key using the startswith command

    example:
    filter_dict_starts_with({"a": 10, "aa": 11, "b": 20}, "a")
    returns: {'a': 10, 'aa': 11}

    :param el: The dict you want to filter
    :param start_with: The string it should start with
    :return: The filtered dict
    """
    return filter_dict(el, lambda key: key.startswith(start_with))


def filter_spaces_start_with(spaces: Dict, start_with: str) -> gym.spaces.Dict:
    """
    Filters a dict and returns the filtered dict as new gym space

    :param spaces: The gym space as dict
    :param start_with: The string the key should start with
    :return: The gym action space
    """
    return gym.spaces.Dict(filter_dict_starts_with(spaces, start_with))


class DummyStructuredEnvironment(Wrapper[MazeEnv], StructuredEnv, StructuredEnvSpacesMixin):
    """
    A structured environment which returns random actions
    """

    def __init__(self, maze_env: MazeEnv):
        Wrapper.__init__(self, maze_env)

        # initialize action space
        self._action_spaces_dict = {
            0: filter_spaces_start_with(maze_env.action_space.spaces, "action_0"),
            1: filter_spaces_start_with(maze_env.action_space.spaces, "action_1")
        }
        self._observation_spaces_dict = {
            0: filter_spaces_start_with(maze_env.observation_space.spaces, "observation_0"),
            1: filter_spaces_start_with(maze_env.observation_space.spaces, "observation_1")
        }

        StructuredEnv.__init__(self)
        self._num_sub_steps = len(self._action_spaces_dict)
        self._sub_step_index = 0
        self.maze_env = maze_env

        self.last_obs = None

    def reset(self) -> Any:
        """Resets the environment and returns the initial state.

        :return: the initial state after resetting.
        """
        self.last_obs = self.env.reset()
        self._sub_step_index = 0
        return filter_dict_starts_with(self.last_obs, 'observation_0')

    def step(self, action) -> Tuple[Dict, float, bool, Optional[Dict]]:
        """Generic sub-step function.

        :return: state, reward, done, info
        """

        sub_step_result = None
        if self._sub_step_index == 0:
            sub_step_result = self._action0(action)
        elif self._sub_step_index == 1:
            sub_step_result = self._action1(action)

        self._sub_step_index = 0 if self._sub_step_index == 1 else 1
        return sub_step_result

    def seed(self, seed: int = None) -> None:
        """Sets the seed for this environment's random number generator(s).

        :param: seed: the seed integer initializing the random number generator.
        """
        self.env.seed(seed)

    def actor_id(self) -> ActorID:
        """
        :return The action id (sub_step_index, not used)
        """
        return ActorID(step_key=self._sub_step_index, agent_id=0)

    def is_actor_done(self) -> bool:
        """Actors are never destroyed in this env."""
        return False

    @property
    def agent_counts_dict(self) -> Dict[StepKeyType, int]:
        """Two-step, single agent env."""
        return {0: 1, 1: 1}

    def close(self) -> None:
        """Performs any necessary cleanup.
        """
        self.env.close()

    @property
    def action_space(self) -> gym.spaces.Dict:
        """The currently active gym action space.
        """
        return self._action_spaces_dict[self._sub_step_index]

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """The currently active gym observation space.
        """
        return self._observation_spaces_dict[self._sub_step_index]

    @property
    def action_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """Override the action spaces according to the introduced sub steps."""
        return self._action_spaces_dict

    @property
    def observation_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """Override the observation spaces according to the introduced sub steps."""
        return self._observation_spaces_dict

    def _action0(self, action) -> Tuple[Dict, float, bool, Optional[Dict[str, np.ndarray]]]:
        """
        Returns the first action

        :return: state, reward, done, info
        """
        # Only the second sub step actually steps the underlying core env
        return filter_dict_starts_with(self.last_obs, 'observation_1'), 1, False, {}

    def _action1(self, action) -> Tuple[Dict, float, bool, Optional[Dict[str, np.ndarray]]]:
        """
        Returns the second action

        :return: state, reward, done, info
        """
        # Only the second sub step actually steps the underlying core env
        self.last_obs, _, _, _ = self.maze_env.step(action)
        return filter_dict_starts_with(self.last_obs, 'observation_0'), 2, False, {}
