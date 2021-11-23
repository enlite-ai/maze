"""Numpy implementation of ES optimizers, based on https://github.com/openai/evolution-strategies-starter"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from maze.core.agent.torch_model import TorchModel
from maze.train.trainers.es.es_utils import get_flat_parameters, set_flat_parameters


class Optimizer(ABC):
    """Abstract baseclass of an optimizer to be used with ES."""

    def __init__(self):
        self.policy: Optional[TorchModel] = None
        self.dim: Optional[int] = None
        self.t: Optional[int] = None

    def setup(self, policy: TorchModel) -> None:
        """Two-stage construction to enable construction from config-files.

        :param policy: ES policy network to optimize
        """
        self.policy = policy
        self.dim = policy.num_params
        self.t = 0

    def update(self, global_gradient: np.ndarray) -> float:
        """Execute one update step.

        :param global_gradient: A flat gradient vector

        :return update ratio = norm(optimizer step) / norm(theta)
        """
        self.t += 1
        step = self._compute_step(global_gradient)
        theta = get_flat_parameters(self.policy)
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        set_flat_parameters(self.policy, theta + step)
        return ratio

    @abstractmethod
    def _compute_step(self, global_gradient: np.ndarray) -> np.ndarray:
        """Compute a single step, to be implemented by the concrete optimizers"""
        raise NotImplementedError
