"""Argument parser for Impala algorithm and training"""

from dataclasses import dataclass

from maze.train.trainers.common.config_classes import AlgorithmConfig
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator


@dataclass
class ImpalaAlgorithmConfig(AlgorithmConfig):
    """Algorithm parameters for Impala."""

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

    queue_out_of_sync_factor: float
    """this factor multiplied by the actor_batch_size gives the size of the queue for
       the agents output collected by the learner. Therefor if the all rollouts computed can be at most
       (queue_out_of_sync_factor + num_agents/actor_batch_size) out of sync with learner policy"""

    actors_batch_size: int
    """number of actors to combine to one batch"""

    num_actors: int
    """number of actors to be run"""

    vtrace_clip_rho_threshold: float
    r"""A scalar float32 tensor with the clipping threshold for importance weights
        (rho) when calculating the baseline targets (vs). rho^bar in the paper. If None, no clipping is applied."""

    vtrace_clip_pg_rho_threshold: float
    r"""A scalar float32 tensor with the clipping threshold on rho_s in
        \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_sfrom_importance_weights)). If None, no clipping is
        applied."""

    # Should be Union[RolloutEvaluator, ConfigType], which Hydra does not support (yet).
    rollout_evaluator: RolloutEvaluator
    """Rollout evaluator."""
