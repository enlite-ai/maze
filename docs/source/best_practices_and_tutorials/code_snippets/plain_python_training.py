""" Rollout of a policy in plain Python. """
import gym
import torch.nn as nn
from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_critic import TorchSharedStateCritic
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.blocks.general.torch_model_block import TorchModelBlock
from maze.train.parallelization.distributed_env.dummy_distributed_env import DummyStructuredDistributedEnv


# Model Setup
# ===========
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.train.trainers.a2c.a2c_trainer import MultiStepA2C
from maze.utils.log_stats_utils import setup_logging


class CartpolePolicyNet(nn.Module):
    """ Simple linear policy net for demonstration purposes """
    def __init__(self, in_features, out_features):
        super(CartpolePolicyNet, self).__init__()
        self.dense = nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features))

    def forward(self, x):
        """ Forward method """
        return self.dense(x)


# Wrapped Policy Model:
class WrappedCartpolePolicyNet(nn.Module):
    """ Wrapper for a model that transforms the network into a Maze. compatible one. """
    def __init__(self, obs_shapes, action_logit_shapes):
        super(WrappedCartpolePolicyNet, self).__init__()
        self.policy_network = CartpolePolicyNet(in_features=obs_shapes[0], out_features=action_logit_shapes[0])

    def forward(self, x_dict):
        logits_dict = {'action': self.policy_network.forward(x_dict['observation'])}
        return logits_dict


class CartpoleValueNet(nn.Module):
    """ Simple linear value net for demonstration purposes """
    def __init__(self, in_features):
        super(CartpoleValueNet, self).__init__()
        self.dense = nn.Sequential(nn.Linear(in_features=in_features, out_features=1))

    def forward(self, x):
        """ Forward method """
        return self.dense(x)

# Wrapped Value Model:
class WrappedCartpoleValueNet(nn.Module):
    """ Wrapper for a model that transforms the network into a Maze. compatible one. """
    def __init__(self, obs_shapes):
        super(WrappedCartpoleValueNet, self).__init__()
        self.value_net = CartpoleValueNet(in_features=obs_shapes[0])

    def forward(self, x_dict):
        """ Forward method. """
        value_dict = {'value': self.value_net.forward(x_dict['observation'])}
        return value_dict

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

# Distributed Environments
# ------------------------
# The factory can now be supplied to one of Maze's distribution classes:
train_envs = DummyStructuredDistributedEnv([cartpole_env_factory for _ in range(2)], logging_prefix="train")
eval_envs = DummyStructuredDistributedEnv([cartpole_env_factory for _ in range(2)], logging_prefix="eval")

# Model Setup
# ===========

# Policy Setup
# ------------
# Instantiate one environment. This will be used for convenient access to observation and action spaces.
env = cartpole_env_factory()
observation_space = env.observation_space
action_space = env.action_space

# Policy Network
# ^^^^^^^^^^^^^^
# Instantiate policy with the correct shapes of observation and action spaces.
policy_net = WrappedCartpolePolicyNet(obs_shapes=observation_space.spaces['observation'].shape,
                                      action_logit_shapes=(action_space.spaces['action'].n,))

maze_wrapped_policy_net = TorchModelBlock(
    in_keys='observation', out_keys='action',
    in_shapes=observation_space.spaces['observation'].shape, in_num_dims=[2],
    out_num_dims=2, net=policy_net)

policy_networks = {0: maze_wrapped_policy_net}

# Policy Distribution
# ^^^^^^^^^^^^^^^^^^^
distribution_mapper = DistributionMapper(action_space=action_space, distribution_mapper_config={})

# Optionally, you can specify a different distribution with the distribution_mapper_config argument. Using a
# Categorical distribution for a discrete action space would be done via
distribution_mapper = DistributionMapper(action_space=action_space, distribution_mapper_config=[{
    "action_space": gym.spaces.Discrete,
    "distribution": "maze.distributions.categorical.CategoricalProbabilityDistribution"
}])

# Instantiating the Policy
# ^^^^^^^^^^^^^^^^^^^^^^^^
torch_policy = TorchPolicy(networks=policy_networks, distribution_mapper=distribution_mapper, device='cpu')


# Value Function Setup
# --------------------

# Value Network
# ^^^^^^^^^^^^^
value_net = WrappedCartpoleValueNet(obs_shapes=observation_space.spaces['observation'].shape)

maze_wrapped_value_net = TorchModelBlock(
    in_keys='observation', out_keys='value',
    in_shapes=observation_space.spaces['observation'].shape, in_num_dims=[2],
    out_num_dims=2, net=value_net)

value_networks = {0: maze_wrapped_value_net}

# Instantiate the Value Function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch_critic = TorchSharedStateCritic(networks=value_networks, num_policies=1, device='cpu')


# Initializing the ActorCritic Model.
# -----------------------------------
actor_critic_model = TorchActorCritic(policy=torch_policy, critic=torch_critic, device='cpu')

# Instantiating the Trainer
# =========================
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
    device='cpu')

a2c_trainer = MultiStepA2C(env=train_envs, eval_env=eval_envs, algorithm_config=algorithm_config,
                           model=actor_critic_model, model_selection=None)

# Train the Agent
# ===============
# Before starting the training, we will enable logging by calling
log_dir = '.'
setup_logging(job_config=None, log_dir=log_dir)

# Now, we can train the agent.
a2c_trainer.train()

# To get an out-of sample estimate of our performance, evaluate on the evaluation envs:
a2c_trainer.evaluate(deterministic=False, repeats=1)
