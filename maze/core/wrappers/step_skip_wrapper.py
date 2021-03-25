""" Implements step skipping as an environment wrapper. """
from collections import defaultdict, deque
from typing import Any, Dict, Tuple, Union, Optional

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import Wrapper, EnvType


class StepSkipWrapper(Wrapper[Union[StructuredEnv, EnvType]]):
    """A step-skip-wrapper providing functionality for skipping n_steps environment steps.
    Options for skipping are: (noop: apply the noop action, sticky: apply the last action again).

    :param env: Environment to wrap.
    :param n_skip_steps: Number of steps that should be skipped.
    :param skip_mode: Skipping action selection mode (noop, sticky).
    """
    SKIPPING_MODES = ['sticky', 'noop']

    def __init__(self, env: Union[StructuredEnvSpacesMixin, MazeEnv], n_skip_steps: int, skip_mode: str):
        super().__init__(env)

        # initialize observation skipping
        self.n_skip_steps = n_skip_steps
        self.skip_mode = skip_mode
        assert self.skip_mode in self.SKIPPING_MODES, \
            f'Skips mode ({self.skip_mode}) has to be one of the following {self.SKIPPING_MODES}'

        assert len(env.action_spaces_dict.keys()) == 1, \
            f'The StepSkipWrapper only supports single step environments.'

    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, Dict[Any, Any]]:
        """Intercept ``BaseEnv.step`` and map observation."""

        # init reward accumulation
        observation, acc_reward, done, info = None, 0, None, None

        # init event buffering
        topics = self.env.core_env.context.event_service.topics
        event_buffer = defaultdict(deque)

        # perform n_steps + 1 steps
        for _ in range(self.n_skip_steps + 1):
            # take env step
            observation, reward, done, info = self.env.step(action)
            # accumulate reward and collect events
            acc_reward += reward
            for key, topic in topics.items():
                event_buffer[key].extend(list(topic.events))
            if done:
                break

        # write buffered events back to topic
        for key, events in event_buffer.items():
            topics[key].events = event_buffer[key]

        return observation, acc_reward, done, info

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType], maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool)\
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Not implemented yet. Note: Some step skipping might be required here as well (depends on the use case)."""
        raise NotImplementedError
