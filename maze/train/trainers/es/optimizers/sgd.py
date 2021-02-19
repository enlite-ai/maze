"""Numpy implementation of ES optimizers, based on https://github.com/openai/evolution-strategies-starter"""
from typing import Optional

import numpy as np
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.train.trainers.es.optimizers.base_optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent with momentum"""

    def __init__(self, step_size: float, momentum: float = 0.9):
        super().__init__()

        # options
        self.step_size, self.momentum = step_size, momentum

        # smoothed gradient, initialized by setup()
        self.v: Optional[np.ndarray] = None

    @override(Optimizer)
    def setup(self, policy: TorchPolicy) -> None:
        """prepare optimizer for training"""
        super().setup(policy)

        self.v = np.zeros(self.dim, dtype=np.float32)

    @override(Optimizer)
    def _compute_step(self, global_gradient: np.ndarray) -> np.ndarray:
        self.v = self.momentum * self.v + (1. - self.momentum) * global_gradient
        step = -self.step_size * self.v
        return step
