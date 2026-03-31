"""Interfaces for vectorised environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.observation_conversion import ObservationType

import numpy as np


class VectorEnv(BaseEnv, ABC):
    """Abstract base class for vectorized environments.

    An instance of this class encapsulates multiple environments under the hood and steps them synchronously.

    Note that actions and observations are handled and returned in a stacked form, i.e. not as a list,
    but as a single action/observation dict where the items have an additional dimension corresponding
    to the number of encapsulated environments (as such a setting is more convenient when working with
    Torch policies). To convert these to/from a list, use the training helpers such as
    func:`maze.train.utils.train_utils.stack_numpy_dict_list` and
    :func:`maze.train.utils.train_utils.unstack_numpy_list_dict`.

    Also note that in structured scenarios, only synchronous environments are supported -- i.e., in each
    sub-step, the actor ID must be the same for all environments.

    :param n_envs: The number of encapsulated environments.
    """

    def __init__(self, n_envs: int):
        self.n_envs = n_envs

    @abstractmethod
    def step(
        self, actions: ActionType
    ) -> tuple[ObservationType, np.ndarray, np.ndarray, np.ndarray, Iterable[dict[Any, Any]]]:
        """Step the environments with the given actions.

        :param actions: the list of actions for the respective envs.
        :return: observations, rewards, terminated list, truncated list, information-dicts all in env-aggregated form.
        """

    @abstractmethod
    def reset(self) -> tuple[Any, dict]:
        """Reset all the environments and return respective observations and info dicts in env-aggregated form.

        :return: observations and info dicts in env-aggregated form.
        """

    @abstractmethod
    def seed(self, seeds: list[Any]) -> None:
        """Sets the seed for this vectorised env's random number generator(s) and its contained parallel envs."""

    def _get_indices(self, indices):
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: (list) the implied list of indices.
        """
        if indices is None:
            indices = range(self.n_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices
