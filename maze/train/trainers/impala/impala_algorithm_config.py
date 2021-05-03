"""Argument parser for Impala algorithm and training"""

from dataclasses import dataclass

from maze.train.trainers.common.config_classes import AlgorithmConfig


@dataclass
class ImpalaAlgorithmConfig(AlgorithmConfig):
    """Algorithm parameters for Impala."""

    epoch_length: int
    """number of updates per epoch"""

    deterministic_eval: bool
    """run evaluation in deterministic mode (argmax-policy)"""

    eval_repeats: int
    """number of evaluation trials"""

    eval_concurrency: int
    """Number of concurrently executed evaluation environments."""

    queue_out_of_sync_factor: float
    """this factor multiplied by the actor_batch_size gives the size of the queue for
       the agents output collected by the learner. Therefor if the all rollouts computed can be at most
       (queue_out_of_sync_factor + num_agents/actor_batch_size) out of sync with learner policy"""

    patience: int
    """number of steps used for early stopping"""

    n_rollout_steps: int = 50
    """number of rolloutstep of each epoch substep"""

    actors_batch_size: int = 2
    """number of actors to combine to one batch"""

    num_actors: int = 2
    """number of actors to be run"""

    lr: float = 0.0002
    """learning rate"""

    gamma: float = 0.98
    """discount factor"""

    policy_loss_coef: float = 1.0
    """coefficient of the policy used in the loss calculation"""

    value_loss_coef: float = 0.5
    """coefficient of the value used in the loss calculation"""

    entropy_coef: float = 0.00025
    """coefficient of the entropy used in the loss calculation"""

    max_grad_norm: float = 0
    """max grad norm for gradient clipping, ignored if value==0"""

    vtrace_clip_rho_threshold: float = 1.0
    r"""A scalar float32 tensor with the clipping threshold for importance weights
        (rho) when calculating the baseline targets (vs). rho^bar in the paper. If None, no clipping is applied."""

    vtrace_clip_pg_rho_threshold: float = 1.0
    r"""A scalar float32 tensor with the clipping threshold on rho_s in
        \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_sfrom_importance_weights)). If None, no clipping is
        applied."""

    reward_clipping: str = "abs_one"
    """the type of reward clipping to be used, options 'abs_one', 'soft_asymmetric', 'None'"""

    device: str = "cpu"
    """Device of the learner (either cpu or cuda). Note that the actors collecting rollouts are always run on CPU."""
