"""Example of how to train a policy with A2C for the gym CartPole environment."""
import torch.nn as nn

from examples.cartpole_custom_net import PolicyNet, ValueNet
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_critic import TorchSharedStateCritic
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.train.trainers.a2c.a2c_trainer import A2C
from maze.utils.log_stats_utils import setup_logging


def main(n_epochs: int) -> None:
    """Trains the cart pole environment with the multi-step a2c implementation.
    """

    # initialize distributed env
    envs = SequentialVectorEnv([lambda: GymMazeEnv(env="CartPole-v0") for _ in range(8)],
                               logging_prefix="train")

    # initialize the env and enable statistics collection
    eval_env = SequentialVectorEnv([lambda: GymMazeEnv(env="CartPole-v0") for _ in range(8)],
                                   logging_prefix="eval")

    # init distribution mapper
    env = GymMazeEnv(env="CartPole-v0")

    # init default distribution mapper
    distribution_mapper = DistributionMapper(action_space=env.action_space, distribution_mapper_config={})

    # initialize policies
    policies = {0: PolicyNet({'observation': (4,)}, {'action': (2,)}, non_lin=nn.Tanh)}

    # initialize critic
    critics = {0: ValueNet({'observation': (4,)})}

    # initialize optimizer
    algorithm_config = A2CAlgorithmConfig(
        n_epochs=n_epochs,
        epoch_length=10,
        deterministic_eval=False,
        eval_repeats=5,
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
        critic=TorchSharedStateCritic(networks=critics, num_critics=1, device=algorithm_config.device,
                                      stack_observations=False),
        device=algorithm_config.device)

    a2c = A2C(rollout_generator=RolloutGenerator(envs), eval_env=eval_env, algorithm_config=algorithm_config,
              model=model, model_selection=None)

    setup_logging(job_config=None)

    # train agent
    a2c.train()

    # final evaluation run
    print("Final Evaluation Run:")
    a2c.evaluate()


if __name__ == "__main__":
    """ main """
    main(n_epochs=1000)
