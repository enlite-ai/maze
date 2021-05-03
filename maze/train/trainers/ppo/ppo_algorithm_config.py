"""Algorithm parameters for multi-step A2C model."""
from dataclasses import dataclass

from maze.train.trainers.common.config_classes import AlgorithmConfig


@dataclass
class PPOAlgorithmConfig(AlgorithmConfig):
    """ Algorithm parameters for multi-step PPO model."""

    n_epochs: int
    """number of epochs to train"""

    epoch_length: int
    """number of updates per epoch"""

    deterministic_eval: bool
    """run evaluation in deterministic mode (argmax-policy)"""

    eval_repeats: int
    """number of evaluation trials"""

    patience: int
    """number of steps used for early stopping"""

    critic_burn_in_epochs: int
    """Number of critic (value function) burn in epochs"""

    n_rollout_steps: int
    """Number of steps taken for each rollout"""

    lr: float
    """learning rate"""

    gamma: float
    """discounting factor"""

    gae_lambda: float
    """bias vs variance trade of factor for GAE"""

    policy_loss_coef: float
    """weight of policy loss"""

    value_loss_coef: float
    """weight of value loss"""

    entropy_coef: float
    """weight of entropy loss"""

    max_grad_norm: float
    """The maximum allowed gradient norm during training"""

    device: str
    """Either "cpu" or "cuda" """

    batch_size: int
    """The batch size used for policy and value updates"""

    n_optimization_epochs: int
    """Number of epochs for for policy and value optimization"""

    clip_range: float
    """Clipping parameter of surrogate loss"""
