"""Algorithm parameters for evolution strategies model."""
from dataclasses import dataclass
from typing import Any
from maze.train.trainers.common.config_classes import AlgorithmConfig


@dataclass
class ESAlgorithmConfig(AlgorithmConfig):
    """
    Algorithm parameters for evolution strategies model.
    Note: Pass 0 to n_epochs to train indefinitely.
    """

    n_rollouts_per_update: int
    """Minimum number of episode rollouts per training iteration (=epoch)."""

    n_timesteps_per_update: int
    """Minimum number of cumulative env steps per training iteration (=epoch).
       The training iteration is only finished, once the given number of episodes
       AND the given number of steps has been reached. One of the two parameters
       can be set to 0."""

    max_steps: int
    """Limit the episode rollouts to a maximum number of steps. Set to 0 to disable this option."""

    optimizer: Any
    """The optimizer to use to update the policy based on the sampled gradient."""

    l2_penalty: float
    """L2 weight regularization coefficient."""

    noise_stddev: float
    """The scaling factor of the random noise applied during training."""
