"""Argument parser for SAC algorithm and training"""

from dataclasses import dataclass
from typing import Union, List, Optional

from maze.core.agent.policy import Policy
from maze.core.utils.factory import ConfigType
from maze.train.trainers.common.config_classes import AlgorithmConfig
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator


@dataclass
class SACAlgorithmConfig(AlgorithmConfig):
    """Algorithm parameters for SAC."""

    n_rollout_steps: int
    """number of rolloutstep of each epoch substep"""

    lr: float
    """learning rate"""

    entropy_coef: float
    """entropy coefficient to use if entropy tuning is set to false (called alpha in the org paper)"""

    gamma: float
    """discount factor"""

    max_grad_norm: float
    """max grad norm for gradient clipping, ignored if value==0"""

    num_actors: int
    """number of actors to be run"""

    batch_size: int
    """batch size to be sampled from the buffer"""

    num_batches_per_iter: int
    """Number of batches to update on in each iteration"""

    tau: float
    """Parameter weighting the soft update of the target network"""

    target_update_interval: int
    """Specify in what intervals to update the target networks"""

    device: str
    """device the learner should work one (ether cpu or cuda)"""

    entropy_tuning: bool
    """Specify whether to tune the entropy in the return computation or used a static value (called alpha tuning in the
       org paper)"""

    target_entropy_multiplier: float
    """Specify an optional multiplier for the target entropy.
    This value is multiplied with the default target entropy computation (called alpha tuning in the paper):
        
        - discrete spaces: target_entropy = target_entropy_multiplier * ( - 0.98 * (-log (1 / cardinality(A)))
        - continues spaces: target_entropy = target_entropy_multiplier * (- dim(A)) (e.g., -6 for HalfCheetah-v1)"""

    entropy_coef_lr: float
    """Learning for entropy tuning"""

    split_rollouts_into_transitions: bool
    """Specify whether all computed rollouts should be split into transitions before processing them"""

    replay_buffer_size: int
    """The size of the replay buffer"""

    initial_buffer_size: int
    """The initial buffer size, where transaction are sampled with the initial sampling policy"""

    initial_sampling_policy: Optional[Union[Policy, ConfigType]]
    """The policy used to initially fill the replay buffer"""

    rollouts_per_iteration: int
    """Number of rollouts collected from the actor in each iteration"""

    n_epochs: int
    """number of epochs to train"""

    epoch_length: int
    """number of updates per epoch"""

    patience: int
    """number of steps used for early stopping"""

    # Should be Union[RolloutEvaluator, ConfigType], which Hydra does not support (yet).
    rollout_evaluator: RolloutEvaluator
    """Rollout evaluator."""
