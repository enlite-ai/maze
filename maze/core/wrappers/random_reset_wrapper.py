""" Implements random resetting as an environment wrapper. """
from typing import Any, Union, Optional, Dict, Tuple

import numpy as np

from maze.core.annotations import override
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import Wrapper, EnvType


class RandomResetWrapper(Wrapper[Union[StructuredEnv, EnvType]]):
    """A wrapper skipping the first few steps by taking random actions.
    This is useful for skipping irrelevant initial parts of a trajectory or for introducing randomness in the training
    process.

    :param env: Environment/wrapper to wrap.
    :param min_skip_steps: Minimum number of steps to skip.
    :param max_skip_steps: Maximum number of steps to skip.
    """

    def __init__(self, env: Union[StructuredEnvSpacesMixin, MazeEnv], min_skip_steps: int, max_skip_steps: int):
        super().__init__(env)
        assert min_skip_steps <= max_skip_steps

        # initialize observation skipping
        self.min_skip_steps = min_skip_steps
        self.max_skip_steps = max_skip_steps

    @override(BaseEnv)
    def reset(self) -> Any:
        """Override BaseEnv.reset to reset the step count.
        """

        # sample number of steps to skip and reset env
        skip_steps = np.random.randint(self.min_skip_steps, self.max_skip_steps + 1)
        obs = self.env.reset()

        # skip steps
        for _ in range(skip_steps):
            action = self.action_space.sample()
            obs, rew, done, info = self.env.step(action)
            assert not done, "Your environment was done during random resetting. This should not happen! " \
                             "Make sure you set a valid number of skipping steps."

        return obs

    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType], maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """This wrapper does not modify observations and actions."""
        return self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)
