"""Interfaces for distributed environments."""
from abc import ABC, abstractmethod
from typing import Iterable, Any, Tuple, Dict

import numpy as np

from maze.core.env.base_env import BaseEnv


class BaseDistributedEnv(BaseEnv, ABC):
    """Abstract base class for distributed environments.

    :param: num_envs: the number of distributed environments.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs

    @abstractmethod
    def step(self, actions: Iterable[Any]
             ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Iterable[Dict[Any, Any]]]:
        """Step the environments with the given actions.

        :param actions: the list of actions for the respective envs.
        :return: observations, rewards, dones, information-dicts all in env-aggregated form.
        """

    @abstractmethod
    def reset(self):
        """Reset all the environments and return respective observations in env-aggregated form.

        :return: observations in env-aggregated form.
        """

    @abstractmethod
    def seed(self, seed: int = None) -> None:
        """Sets the seed for this distributed env's random number generator(s) and its contained parallel envs.
        """

    def _get_indices(self, indices):
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: (list) the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices
