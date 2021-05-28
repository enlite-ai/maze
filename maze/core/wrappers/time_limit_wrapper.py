"""Implementation of a wrapper to limit the environment step count, based on gym TimeLimit."""

from typing import TypeVar, Union, Dict, Tuple, Any, Optional

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.time_env_mixin import TimeEnvMixin
from maze.core.wrappers.wrapper import Wrapper, EnvType


class TimeLimitWrapper(Wrapper[Union[BaseEnv, EnvType]], BaseEnv):
    """Wrapper to limit the environment step count, equivalent to gym.wrappers.time_limit.

    Additionally to the gym wrapper, this one supports adjusting the limit after construction.

    :param env: The environment to wrap.
    :param max_episode_steps: The maximum number of steps to take. If 0, the step limit is disabled.
    """

    def __init__(self, env: BaseEnv, max_episode_steps: Optional[int] = None):
        """"private" constructor, the preferred way of constructing this class is by calling :method:`wrap`
        """
        super().__init__(env)

        # attribute declaration
        self._max_episode_steps: int = 0
        self._elapsed_steps = None

        self.set_max_episode_steps(max_episode_steps)

    T = TypeVar("T")

    def set_max_episode_steps(self, max_episode_steps: int) -> None:
        """Set the step limit.

        :param max_episode_steps: The environment step() function sets the done flag if this step limit is reached.
                                  If 0, the step limit is disabled.

        """
        # gym spec
        spec = getattr(self.env, "spec", None)

        if not max_episode_steps and spec is not None:
            max_episode_steps = spec.max_episode_steps
        if spec is not None:
            spec.max_episode_steps = max_episode_steps

        self._max_episode_steps = max_episode_steps

    @override(BaseEnv)
    def step(self, action: Any) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Override BaseEnv.step and set done if the step limit is reached.
        """
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)

        # load time from the environment or increment own step count it the environment does not manage
        # its own time
        if isinstance(self.env, TimeEnvMixin):
            self._elapsed_steps = self.env.get_env_time()
        else:
            self._elapsed_steps += 1

        if self._max_episode_steps and self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    @override(BaseEnv)
    def reset(self) -> Any:
        """Override BaseEnv.reset to reset the step count.
        """
        self._elapsed_steps = 0
        return self.env.reset()

    @override(BaseEnv)
    def seed(self, seed: int) -> None:
        """forward call to the inner env
        """
        return self.env.seed(seed=seed)

    @override(BaseEnv)
    def close(self) -> None:
        """forward call to the inner env
        """
        return self.env.close()

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType], maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool)\
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """This wrapper does not modify observations and actions."""
        return self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'TimeLimitWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self._elapsed_steps = env._elapsed_steps
        self.env.clone_from(env)
