"""Numpy implementation of ES optimizers, based on https://github.com/openai/evolution-strategies-starter"""
from typing import Optional

import numpy as np
from maze.core.agent.torch_policy import TorchPolicy
from maze.train.trainers.es.optimizers.base_optimizer import Optimizer


class Adam(Optimizer):
    """Adam optimizer"""

    def __init__(self, step_size, beta1=0.9, beta2=0.999, epsilon=1e-08):
        super().__init__()

        # options
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # initialized by setup()
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None

    def setup(self, policy: TorchPolicy) -> None:
        """prepare optimizer for training"""
        super().setup(policy)

        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, global_gradient: np.ndarray) -> np.ndarray:
        a = self.step_size * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * global_gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (global_gradient * global_gradient)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
