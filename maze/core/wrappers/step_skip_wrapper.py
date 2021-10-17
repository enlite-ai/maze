""" Implements step skipping as an environment wrapper. """
import copy
from typing import Any, Dict, Tuple, Union, Optional

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import Wrapper, EnvType


class StepSkipWrapper(Wrapper[Union[StructuredEnv, EnvType]]):
    """A step-skip-wrapper providing functionality for skipping n_steps environment steps.
    Options for skipping are: (noop: apply the noop action, sticky: apply the last action again).

    :param env: Environment to wrap.
    :param n_steps: Total number of steps that should be taken (e.g., n_steps = 1 + n_skip_steps).
    :param skip_mode: Skipping action selection mode (noop, sticky).
    """
    SKIPPING_MODES = ['sticky', 'noop']

    def __init__(self, env: Union[StructuredEnvSpacesMixin, MazeEnv], n_steps: int, skip_mode: str):
        super().__init__(env, keep_inner_hooks=True)

        # initialize observation skipping
        self.n_steps = n_steps
        self.skip_mode = skip_mode
        assert self.skip_mode in self.SKIPPING_MODES, \
            f'Skips mode ({self.skip_mode}) has to be one of the following {self.SKIPPING_MODES}'

        # init action recording
        self._step_actions = dict()
        self._steps_done = 0

    def _record_action(self, action: ActionType) -> None:
        """Record the current action for later replay.

        :param action: The action the agent wants to execute.
        """
        step_key = self.actor_id()[0]

        if self.skip_mode == 'sticky':
            self._step_actions[step_key] = copy.deepcopy(action)
        elif self.skip_mode == 'noop':
            self._step_actions[step_key] = self.env.action_conversion.noop_action()
            assert self._step_actions[step_key] is not None, \
                'noop action not defined in the action_conversion interface'

    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, Dict[Any, Any]]:
        """Intercept ``BaseEnv.step`` and map observation."""

        # record the actions given until one flat step finished
        self._record_action(action)
        # execute step
        self._steps_done += 1
        observation, reward, done, info = self.env.step(action)
        # prepare reward accumulation
        acc_reward = reward

        # skipping is finished if the env is done
        if done or self._steps_done >= self.n_steps:
            self._reset_recording()
            return observation, acc_reward, done, info

        # check if all sub-steps have been executed once
        if self.actor_id()[0] != 0:
            # skipping not yet possible, proceed to next sub-step
            return observation, acc_reward, done, info

        # continue with replay
        while self._steps_done < self.n_steps:
            self._steps_done += 1

            # actual skipping: replay recorded actions
            step_key = self.actor_id()[0]
            action = self._step_actions[step_key]
            observation, reward, done, info = self.env.step(action)
            # accumulate reward and collect events
            acc_reward += reward
            if done:
                break

        # skipping finished
        self._reset_recording()
        return observation, acc_reward, done, info

    def _reset_recording(self) -> None:
        """reset the action recording"""
        self._step_actions = dict()
        self._steps_done = 0

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Not implemented yet. Note: Some step skipping might be required here as well (depends on the use case)."""
        raise NotImplementedError

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'StepSkipWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self._step_actions = copy.deepcopy(env._step_actions)
        self._steps_done = env._steps_done
        self.env.clone_from(env)
