"""
Training and rollout of a policy in plain Python.
"""

from typing import Sequence, Dict

import gym
import torch
import torch.nn as nn

from maze.api.utils import RunMode
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv

from maze.utils.log_stats_utils import setup_logging

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.train.trainers.a2c.a2c_trainer import A2C
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.perception.models.critics.shared_state_critic_composer import SharedStateCriticComposer
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.api import run_context
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.blocks.general.torch_model_block import TorchModelBlock
from maze.perception.models.policies import ProbabilisticPolicyComposer


def cartpole_env_factory() -> GymMazeEnv:
    """ Env factory for the cartpole MazeEnv """
    # Registered gym environments can be instantiated first and then provided to GymMazeEnv:
    cartpole_env = gym.make("CartPole-v0")
    maze_env = GymMazeEnv(env=cartpole_env)

    # Another possibility is to supply the gym env string to GymMazeEnv directly:
    # maze_env = GymMazeEnv(env="CartPole-v0")

    return maze_env


class CartpolePolicyNet(nn.Module):
    """ Simple linear policy net for demonstration purposes. """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], action_logit_shapes: Dict[str, Sequence[int]]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                in_features=obs_shapes['observation'][0],
                out_features=action_logit_shapes['action'][0]
            )
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


def train(n_epochs: int) -> int:
    """
    Trains agent in pure Python.

    :param n_epochs: Number of epochs to train.

    :return: 0 if successful.

    """

    # Environment setup
    # -----------------

    env = cartpole_env_factory()

    # Algorithm setup
    # ---------------

    algorithm_config = A2CAlgorithmConfig(
        n_epochs=5,
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
        device='cpu'
    )

    # Custom model setup
    # ------------------

    # Policy customization
    # ^^^^^^^^^^^^^^^^^^^^

    # Policy network.
    policy_net = CartpolePolicyNet(
        obs_shapes={'observation': env.observation_space.spaces['observation'].shape},
        action_logit_shapes={'action': (env.action_space.spaces['action'].n,)}
    )
    policy_networks = [policy_net]

    # Policy distribution.
    distribution_mapper = DistributionMapper(action_space=env.action_space, distribution_mapper_config={})

    # Policy composer.
    policy_composer = ProbabilisticPolicyComposer(
        action_spaces_dict=env.action_spaces_dict,
        observation_spaces_dict=env.observation_spaces_dict,
        # Derive distribution from environment's action space.
        distribution_mapper=distribution_mapper,
        networks=policy_networks,
        # We have only one agent and network, thus this is an empty list.
        substeps_with_separate_agent_nets=[],
        # We have only one step and one agent.
        agent_counts_dict={0: 1}
    )

    # Critic customization
    # ^^^^^^^^^^^^^^^^^^^^

    # Value networks.
    value_networks = {
        0: TorchModelBlock(
            in_keys='observation', out_keys='value',
            in_shapes=env.observation_space.spaces['observation'].shape,
            in_num_dims=[2],
            out_num_dims=2,
            net=CartpoleValueNet({'observation': env.observation_space.spaces['observation'].shape})
        )
    }

    # Critic composer.
    critic_composer = SharedStateCriticComposer(
        observation_spaces_dict=env.observation_spaces_dict,
        agent_counts_dict={0: 1},
        networks=value_networks,
        stack_observations=True
    )

    # Training
    # ^^^^^^^^

    rc = run_context.RunContext(
        env=cartpole_env_factory,
        algorithm=algorithm_config,
        policy=policy_composer,
        critic=critic_composer,
        runner="dev"
    )
    rc.train(n_epochs=n_epochs)

    # Distributed training
    # ^^^^^^^^^^^^^^^^^^^^

    rc = run_context.RunContext(
        env=cartpole_env_factory,
        algorithm=algorithm_config,
        policy=policy_composer,
        critic=critic_composer,
        runner="local"
    )
    rc.train(n_epochs=n_epochs)

    # Evaluation
    # ^^^^^^^^^^

    print("-----------------")
    evaluator = RolloutEvaluator(
        eval_env=LogStatsWrapper.wrap(cartpole_env_factory(), logging_prefix="eval"),
        n_episodes=1,
        model_selection=None
    )
    evaluator.evaluate(rc.policy)

    return 0


if __name__ == '__main__':
    train(n_epochs=1)
