"""Contains unit tests for ppo."""

import torch.nn as nn

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_critic import TorchSharedStateCritic
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.parallelization.vector_env.subproc_vector_env import SubprocVectorEnv
from maze.train.trainers.ppo.ppo_algorithm_config import PPOAlgorithmConfig
from maze.train.trainers.ppo.ppo_trainer import PPO
from maze.perception.models.built_in.flatten_concat import FlattenConcatPolicyNet, FlattenConcatStateValueNet


def train_function(n_epochs: int, distributed_env_cls) -> PPO:
    """Trains the cart pole environment with the multi-step ppo implementation.
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

    # algorithm config
    algorithm_config = PPOAlgorithmConfig(
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
        max_grad_norm=1.0,
        device="cpu",
        batch_size=10,
        n_optimization_epochs=1,
        clip_range=0.2)

    # initialize actor critic model
    model = TorchActorCritic(
        policy=TorchPolicy(networks=policies, distribution_mapper=distribution_mapper, device=algorithm_config.device),
        critic=TorchSharedStateCritic(networks=critics, num_policies=1, device=algorithm_config.device,
                                      stack_observations=False),
        device=algorithm_config.device)

    ppo = PPO(env=envs, algorithm_config=algorithm_config, eval_env=eval_env, model=model,
              model_selection=None)

    # train agent
    ppo.train()

    return ppo


def test_ppo_multi_step():
    """ ppo unit tests """
    ppo = train_function(n_epochs=2, distributed_env_cls=SequentialVectorEnv)
    assert isinstance(ppo, PPO)

    ppo = train_function(n_epochs=2, distributed_env_cls=SequentialVectorEnv)
    assert isinstance(ppo, PPO)


def test_ppo_multi_step_distributed():
    """ ppo unit tests """
    ppo = train_function(n_epochs=2, distributed_env_cls=SubprocVectorEnv)
    assert isinstance(ppo, PPO)
