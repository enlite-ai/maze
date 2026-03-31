"""Translate gym-style actions to :obj:`~maze.core.env.core_env.CoreEnv` specific MazeActions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType

import numpy as np
import torch
from gymnasium import spaces

ActionType = dict[str, int | np.ndarray]
TorchActionType = dict[str, torch.Tensor]


class ActionConversionInterface(ABC):
    """Interface specifying the conversion of agent actions to actual environment MazeActions."""

    @abstractmethod
    def space_to_maze(self, action: dict[str, np.ndarray], maze_state: MazeStateType) -> MazeActionType:
        """Converts agent action to environment MazeAction.

        :param action: the agent action.
        :param maze_state: the environment state.
        :return: the environment MazeAction.
        """

    def maze_to_space(self, maze_action: MazeActionType) -> dict[str, np.ndarray]:  # noqa: B027
        """Converts environment MazeAction to agent action.

        :param maze_action: the environment MazeAction.
        :return: the agent action.
        """

    @abstractmethod
    def space(self) -> spaces.Dict:
        """Returns respective gym action space."""

    def noop_action(self):  # noqa: B027
        """Converts environment MazeAction to agent action.

        :return: the noop action.
        """

    def create_action_hash(self, action: ActionType) -> int | str:  # noqa: B027
        """Calculate a hash of the given action. Can be used for identifying and de-duplicating
        same actions when creating/evaluating action scenarios.

        :param action: Action to hash
        :return: Hash of a given action, either as a string, or integer (should be of the same type for all actions)
        """
