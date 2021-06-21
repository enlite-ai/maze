"""Algorithm parameters for multi-step A2C model."""
from dataclasses import dataclass

from maze.train.trainers.common.config_classes import AlgorithmConfig
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator


@dataclass
class A2CAlgorithmConfig(AlgorithmConfig):
    """ Algorithm parameters for multi-step A2C model."""

    n_epochs: int
    """number of epochs to train"""

    epoch_length: int
    """number of updates per epoch"""

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

    # Should be Union[RolloutEvaluator, ConfigType], which Hydra does not support (yet).
    rollout_evaluator: RolloutEvaluator
    """Rollout evaluator."""
