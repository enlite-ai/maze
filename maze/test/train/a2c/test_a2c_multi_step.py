"""Contains unit tests for a2c."""

import torch.nn as nn
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_critic import TorchSharedStateCritic

from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.parallelization.vector_env.subproc_vector_env import SubprocVectorEnv
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.train.trainers.a2c.a2c_trainer import A2C
from maze.perception.models.built_in.flatten_concat import FlattenConcatPolicyNet, FlattenConcatStateValueNet


def train_function(n_epochs: int, distributed_env_cls) -> A2C:
    """Trains the cart pole environment with the multi-step a2c implementation.
    """

    # initialize distributed env
    envs = distributed_env_cls([lambda: GymMazeEnv(env="CartPole-v0") for _ in range(2)])

    # initialize the env and enable statistics collection
    eval_env = distributed_env_cls([lambda: GymMazeEnv(env="CartPole-v0") for _ in range(2)],
                                   logging_prefix='eval')

    # init distribution mapper
    env = GymMazeEnv(env="CartPole-v0")
    distribution_mapper = DistributionMapper(action_space=env.action_space, distribution_mapper_config={})

    # initialize policies
    policies = {0: FlattenConcatPolicyNet({'observation': (4,)}, {'action': (2,)}, hidden_units=[16], non_lin=nn.Tanh)}

    # initialize critic
    critics = {0: FlattenConcatStateValueNet({'observation': (4,)}, hidden_units=[16], non_lin=nn.Tanh)}

    # algorithm configuration
    algorithm_config = A2CAlgorithmConfig(
        n_epochs=n_epochs,
        epoch_length=2,
        deterministic_eval=False,
        eval_repeats=2,
        patience=10,
        critic_burn_in_epochs=0,
        n_rollout_steps=20,
        lr=0.0005,
        gamma=0.98,
        gae_lambda=1.0,
        policy_loss_coef=1.0,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.0,
        device="cpu")

    # initialize actor critic model
    model = TorchActorCritic(
        policy=TorchPolicy(networks=policies, distribution_mapper=distribution_mapper, device=algorithm_config.device),
        critic=TorchSharedStateCritic(networks=critics, num_policies=1, device=algorithm_config.device,
                                      stack_observations=False),
        device=algorithm_config.device)

    a2c = A2C(env=envs, algorithm_config=algorithm_config, eval_env=eval_env, model=model,
              model_selection=None)

    # train agent
    a2c.train()

    return a2c


def test_a2c_multi_step():
    """ A2C unit tests """
    a2c = train_function(n_epochs=2, distributed_env_cls=SequentialVectorEnv)
    assert isinstance(a2c, A2C)

    a2c = train_function(n_epochs=2, distributed_env_cls=SequentialVectorEnv)
    assert isinstance(a2c, A2C)


def test_a2c_multi_step_distributed():
    """ A2C unit tests """
    a2c = train_function(n_epochs=2, distributed_env_cls=SubprocVectorEnv)
    assert isinstance(a2c, A2C)
