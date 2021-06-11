""" Rollout of a policy in plain Python. """

from typing import Dict, Sequence

import gym
import torch
import torch.nn as nn

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_critic import TorchSharedStateCritic
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.blocks.general.torch_model_block import TorchModelBlock
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.train.trainers.a2c.a2c_trainer import A2C
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.utils.log_stats_utils import setup_logging


# Environment Setup
# =================

# Environment Factory
# -------------------
# Define environment factory
def cartpole_env_factory():
    """ Env factory for the cartpole MazeEnv """
    # Registered gym environments can be instantiated first and then provided to GymMazeEnv:
    cartpole_env = gym.make("CartPole-v0")
    maze_env = GymMazeEnv(env=cartpole_env)

    # Another possibility is to supply the gym env string to GymMazeEnv directly:
    maze_env = GymMazeEnv(env="CartPole-v0")

    return maze_env


# Model Setup
# ===========
# Policy Network
# --------------
class CartpolePolicyNet(nn.Module):
    """ Simple linear policy net for demonstration purposes. """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], action_logit_shapes: Dict[str, Sequence[int]]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=obs_shapes['observation'][0],
                      out_features=action_logit_shapes['action'][0])
        )

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Since x_dict has to be a dictionary in Maze, we extract the input for the network.
        x = x_dict['observation']

        # Do the forward pass.
        logits = self.net(x)

        # Since the return value has to be a dict again, put the forward pass result into a dict with the
        # correct key.
        logits_dict = {'action': logits}

        return logits_dict


# Value Network
# -------------
class CartpoleValueNet(nn.Module):
    """ Simple linear value net for demonstration purposes. """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]]):
        super().__init__()
        self.value_net = nn.Sequential(nn.Linear(in_features=obs_shapes['observation'][0], out_features=1))

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ Forward method. """
        # The same as for the policy can be said about the value net. Inputs and outputs have to be dicts.
        x = x_dict['observation']

        value = self.value_net(x)

        value_dict = {'value': value}
        return value_dict


def train(n_epochs):
    # Instantiate one environment. This will be used for convenient access to observation
    # and action spaces.
    env = cartpole_env_factory()
    observation_space = env.observation_space
    action_space = env.action_space

    # Policy Setup
    # ------------

    # Policy Network
    # ^^^^^^^^^^^^^^
    # Instantiate policy with the correct shapes of observation and action spaces.
    policy_net = CartpolePolicyNet(
        obs_shapes={'observation': observation_space.spaces['observation'].shape},
        action_logit_shapes={'action': (action_space.spaces['action'].n,)})

    maze_wrapped_policy_net = TorchModelBlock(
        in_keys='observation', out_keys='action',
        in_shapes=observation_space.spaces['observation'].shape, in_num_dims=[2],
        out_num_dims=2, net=policy_net)

    policy_networks = {0: maze_wrapped_policy_net}

    # Policy Distribution
    # ^^^^^^^^^^^^^^^^^^^
    distribution_mapper = DistributionMapper(
        action_space=action_space,
        distribution_mapper_config={})

    # Optionally, you can specify a different distribution with the distribution_mapper_config argument. Using a
    # Categorical distribution for a discrete action space would be done via
    distribution_mapper = DistributionMapper(
        action_space=action_space,
        distribution_mapper_config=[{
            "action_space": gym.spaces.Discrete,
            "distribution": "maze.distributions.categorical.CategoricalProbabilityDistribution"}])

    # Instantiating the Policy
    # ^^^^^^^^^^^^^^^^^^^^^^^^
    torch_policy = TorchPolicy(networks=policy_networks, distribution_mapper=distribution_mapper, device='cpu')

    # Value Function Setup
    # --------------------

    # Value Network
    # ^^^^^^^^^^^^^
    value_net = CartpoleValueNet(obs_shapes={'observation': observation_space.spaces['observation'].shape})

    maze_wrapped_value_net = TorchModelBlock(
        in_keys='observation', out_keys='value',
        in_shapes=observation_space.spaces['observation'].shape, in_num_dims=[2],
        out_num_dims=2, net=value_net)

    value_networks = {0: maze_wrapped_value_net}

    # Instantiate the Value Function
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    torch_critic = TorchSharedStateCritic(networks=value_networks, obs_spaces_dict=env.observation_spaces_dict,
                                          device='cpu', stack_observations=False)

    # Initializing the ActorCritic Model.
    # -----------------------------------
    actor_critic_model = TorchActorCritic(policy=torch_policy, critic=torch_critic, device='cpu')

    # Instantiating the Trainer
    # =========================
    algorithm_config = A2CAlgorithmConfig(
        n_epochs=n_epochs,
        epoch_length=25,
        deterministic_eval=False,
        eval_repeats=2,
        patience=15,
        critic_burn_in_epochs=0,
        n_rollout_steps=100,
        lr=0.0005,
        gamma=0.98,
        gae_lambda=1.0,
        policy_loss_coef=1.0,
        value_loss_coef=0.5,
        entropy_coef=0.00025,
        max_grad_norm=0.0,
        device='cpu')

    # Distributed Environments
    # ------------------------
    # In order to use the distributed trainers, the previously created env factory is supplied to one of Maze's
    # distribution classes:
    train_envs = SequentialVectorEnv([cartpole_env_factory for _ in range(2)], logging_prefix="train")
    eval_envs = SequentialVectorEnv([cartpole_env_factory for _ in range(2)], logging_prefix="eval")

    # Initialize best model selection.
    model_selection = BestModelSelection(dump_file="params.pt", model=actor_critic_model,
                                         dump_interval=None)

    a2c_trainer = A2C(rollout_generator=RolloutGenerator(train_envs), eval_env=eval_envs,
                      algorithm_config=algorithm_config, model=actor_critic_model, model_selection=model_selection)

    # Train the Agent
    # ===============
    # Before starting the training, we will enable logging by calling
    log_dir = '.'
    setup_logging(job_config=None, log_dir=log_dir)

    # Now, we can train the agent.
    a2c_trainer.train()

    return 0


if __name__ == '__main__':
    train(n_epochs=5)
