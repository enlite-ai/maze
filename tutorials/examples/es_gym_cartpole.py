"""Example of how to train a policy with ES for the gym CartPole environment."""

import torch.nn as nn

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.utils.config_utils import list_to_dict
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.perception_utils import observation_spaces_to_in_shapes
from maze.train.trainers.es.distributed.es_dummy_distributed_rollouts import ESDummyDistributedRollouts
from maze.train.trainers.es.es_algorithm_config import ESAlgorithmConfig
from maze.train.trainers.es.es_shared_noise_table import SharedNoiseTable
from maze.train.trainers.es.es_trainer import ESTrainer
from maze.train.trainers.es.optimizers.adam import Adam
from maze.utils.log_stats_utils import setup_logging

from examples.cartpole_custom_net import PolicyNet


def main(max_epochs) -> None:
    """Trains the cart pole environment with the ES implementation.
    """

    env = GymMazeEnv(env="CartPole-v0")
    distribution_mapper = DistributionMapper(action_space=env.action_space, distribution_mapper_config={})

    obs_shapes = observation_spaces_to_in_shapes(env.observation_spaces_dict)
    action_shapes = {step_key: {action_head: distribution_mapper.required_logits_shape(action_head)
                                for action_head in env.action_spaces_dict[step_key].spaces.keys()}
                     for step_key in env.action_spaces_dict.keys()}

    # initialize policies
    policies = [PolicyNet(obs_shapes=obs_shapes[0],
                          action_logits_shapes=action_shapes[0],
                          non_lin=nn.SELU)]

    # initialize optimizer
    policy = TorchPolicy(networks=list_to_dict(policies), separated_agent_networks=False,
                         distribution_mapper=distribution_mapper, device="cpu")

    shared_noise = SharedNoiseTable(count=1_000_000)

    algorithm_config = ESAlgorithmConfig(
        n_rollouts_per_update=100,
        n_timesteps_per_update=0,
        max_steps=0,
        optimizer=Adam(step_size=0.01),
        l2_penalty=0.005,
        noise_stddev=0.02,
        max_epochs=max_epochs)

    trainer = ESTrainer(algorithm_config=algorithm_config,
                        policy=policy,
                        shared_noise=shared_noise,
                        normalization_stats=None)

    setup_logging(job_config=None)

    # run with pseudo-distribution, without worker processes
    trainer.train(ESDummyDistributedRollouts(env=env,
                                             n_eval_rollouts=10,
                                             shared_noise=shared_noise), model_selection=None)


if __name__ == "__main__":
    """ main """
    main(max_epochs=100)
