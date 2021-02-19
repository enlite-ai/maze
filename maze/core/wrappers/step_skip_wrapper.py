""" Implements step skipping as an environment wrapper. """
from typing import Any, Dict, Tuple, Union, Optional

from maze.core.annotations import override
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import Wrapper, EnvType


class StepSkipWrapper(Wrapper[Union[StructuredEnv, EnvType]]):
    """A step-skip-wrapper providing functionality for:
        - skipping n_steps
        - specify what action is provided when skipping a step
            - noop: apply the noop action
            - sticky: apply the last action again

    :param env: Environment/wrapper to wrap.
    :param n_steps: int specifying how many steps should be skipped.
    :param skip_mode: str specifying the mode of skipping.
            options: noop, sticky
    """
    SKIPPING_MODES = ['sticky', 'noop']

    def __init__(self, env: Union[StructuredEnvSpacesMixin, MazeEnv], n_steps: int, skip_mode: str):
        super().__init__(env)

        # initialize observation skipping
        self.n_steps = n_steps
        self.skip_mode = skip_mode
        assert self.skip_mode in self.SKIPPING_MODES, f'Skips mode ({self.skip_mode}) has to be one of the ' \
                                                      f'following {self.SKIPPING_MODES}'
        self._step_actions = dict()
        self._record_actions = True

    def _record_action(self, action) -> None:
        """Record the current action for later use

        :param action: The action the agent wants to execute
        """
        if self.skip_mode == 'sticky':
            self._step_actions[self.actor_id()[0]] = action
        elif self.skip_mode == 'noop':
            self._step_actions[self.actor_id()[0]] = self.env.action_conversion.noop_action()
            assert self._step_actions[self.actor_id()[0]] is not None, 'Noop action not defined in the ' \
                                                                       'action_conversion interface'

    def step(self, action) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Intercept ``BaseEnv.step`` and map observation."""
        if self.n_steps == 0:
            # if n_steps is 0 execute steps normally
            observation, reward, done, info = self.env.step(action)

            return observation, reward, done, info
        elif self._record_actions:
            # If record actions is true record the actions given until one flat step finished
            self._record_action(action)
            observation, reward, done, info = self.env.step(action)
            if self.actor_id() == 0 and len(self._step_actions) > 0:
                self._record_actions = False

            return observation, reward, done, info
        else:
            # Once all sub actions are recorded perform the skipping
            for i in range(self.n_steps):
                for j in range(len(self._step_actions)):
                    observation, reward, done, info = self.env.step(self._step_actions[self.actor_id()[0]])
                    if done:
                        self._record_actions = True
                        self._step_actions = dict()
                        return observation, reward, done, info
            self._record_actions = True
            self._step_actions = dict()
            return observation, reward, done, info

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType], maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool)\
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Not implemented yet. Note: Some step skipping might be required here as well (depends on the use case)."""
        raise NotImplementedError
