"""Algorithm parameters for behavioral cloning."""

from dataclasses import dataclass
from typing import Any

from maze.train.trainers.common.config_classes import AlgorithmConfig


@dataclass
class BCAlgorithmConfig(AlgorithmConfig):
    """Algorithm parameters for behavioral cloning."""

    device: str
    """Either "cpu" or "cuda" """

    batch_size: int
    """Batch size for training"""

    n_eval_workers: int
    """Number of workers to perform evaluation runs in. If set to 1, evaluation is performed in the main process."""

    validation_percentage: float
    """Percentage of the data used for validation."""

    n_epochs: int
    """number of epochs to train"""
    
    eval_every_k_iterations: int
    """Number of iterations after which to run evaluation (in addition to evaluations at the end of
       each epoch, which are run automatically). If set to None, evaluations will run on epoch end only."""

    n_eval_episodes: int
    """Number of episodes to run during each evaluation rollout"""

    max_episode_steps: int
    """Max number of steps per episode to run during each evaluation rollout"""

    optimizer: Any
    """The optimizer to use to update the policy."""
