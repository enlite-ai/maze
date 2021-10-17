from typing import Tuple

import torch.nn as nn

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.structured_env import StructuredEnv
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.models.built_in.flatten_concat import FlattenConcatPolicyNet
from maze.train.trainers.es.distributed.es_dummy_distributed_rollouts import ESDummyDistributedRollouts
from maze.train.trainers.es.es_algorithm_config import ESAlgorithmConfig
from maze.train.trainers.es.es_shared_noise_table import SharedNoiseTable
from maze.train.trainers.es.es_trainer import ESTrainer
from maze.train.trainers.es.optimizers.adam import Adam


def train_setup(n_epochs: int) -> Tuple[TorchPolicy, StructuredEnv, ESTrainer]:
    """Trains the cart pole environment with the multi-step a2c implementation.
    """

    # initialize distributed env
    env = GymMazeEnv(env="CartPole-v0")

    # initialize distribution mapper
    distribution_mapper = DistributionMapper(action_space=env.action_space, distribution_mapper_config={})

    # initialize policies
    policies = {0: FlattenConcatPolicyNet({'observation': (4,)}, {'action': (2,)}, hidden_units=[16], non_lin=nn.Tanh)}

    # initialize optimizer
    policy = TorchPolicy(networks=policies, distribution_mapper=distribution_mapper, device="cpu")

    # reduce the noise table size to speed up testing
    shared_noise = SharedNoiseTable(count=1_000_000)

    algorithm_config = ESAlgorithmConfig(
        n_rollouts_per_update=100,
        n_timesteps_per_update=0,
        max_steps=0,
        optimizer=Adam(step_size=0.01),
        l2_penalty=0.005,
        noise_stddev=0.02,
        n_epochs=n_epochs
    )

    # train agent
    trainer = ESTrainer(algorithm_config=algorithm_config,
                        shared_noise=shared_noise,
                        policy=policy,
                        normalization_stats=None)

    return policy, env, trainer


def test_es():
    policy, env, trainer = train_setup(n_epochs=2)

    trainer.train(
        ESDummyDistributedRollouts(env=env, n_eval_rollouts=2, shared_noise=trainer.shared_noise,
                                   agent_instance_seed=1234), model_selection=None)
