""" Rollout of a policy in plain Python. """
import gym
import torch.nn as nn
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.blocks.general.torch_model_block import TorchModelBlock
from maze.train.parallelization.distributed_env.dummy_distributed_env import DummyStructuredDistributedEnv


# Model Setup
# ===========
class CartpolePolicyNet(nn.Module):
    """ Simple linear policy net for demonstration purposes """
    def __init__(self, in_features, out_features):
        super(CartpolePolicyNet, self).__init__()
        self.dense = nn.Sequential(nn.Linear(in_features=in_features, out_features=out_features))

    def forward(self, x):
        """ Forward method """
        return self.dense(x)


class CartpoleValueNet(nn.Module):
    """ Simple linear value net for demonstration purposes """
    def __init__(self, in_features):
        super(CartpoleValueNet, self).__init__()
        self.dense = nn.Sequential(nn.Linear(in_features=in_features, out_features=1))

    def forward(self, x):
        """ Forward method """
        return self.dense(x)


# Environment Setup
# =================

# Environment Factory
# -------------------
# First, we will prepare our environment for use with Maze. In order to use Maze's parallelization capabilities, it
# is necessary to define a factory function that returns a MazeEnv of your environment. This is easily done for
# gym environments:

# Define environment factory
def cartpole_env_factory():
    """ Env factory for the cartpole MazeEnv """
    # Registered gym environments can be instantiated first and then provided to GymMazeEnv:
    cartpole_env = gym.make("CartPole-v0")
    maze_env = GymMazeEnv(env=cartpole_env)

    # Another possibility is to supply the gym env string to GymMazeEnv directly:
    maze_env = GymMazeEnv(env="CartPole-v0")

    return maze_env

# If you have your own environment you must transform it into a MazeEnv yourself, as is shown in #todo link
# and have your factory return that.


# Distributed Environments
# ------------------------
# The factory can now be supplied to one of Maze's distribution classes:
train_envs = DummyStructuredDistributedEnv([cartpole_env_factory for _ in range(2)], logging_prefix="train")
eval_envs = DummyStructuredDistributedEnv([cartpole_env_factory for _ in range(2)], logging_prefix="eval")

# Policy Setup
# ===========
# For a policy, we need a parametrization for the policy (provided by the policy network), a probability distribution
# and a sampling mechanism to sample from this distribution. We will subsequently define and instantiate each of these.

# Model Setup
# -----------
# Now that the environment setup is done, let us define the policy and value networks that will be used. We will not
# re-use the networks that were introduced in # todo: link to custom-models section
# as they already adhere to the maze model interface. Here, we would like to show how to transform any models that
# you already have to the necessary Maze interface.

# Instantiate one environment. This will be used for convenient access to observation and action spaces.
env = cartpole_env_factory()
observation_space = env.observation_space
action_space = env.action_space

# Instantiate policy with the correct shapes of observation and action spaces.
policy_net = CartpolePolicyNet(in_features=observation_space.spaces['observation'].shape[0],
                               out_features=action_space.spaces['action'].n)
value_net = CartpoleValueNet(in_features=observation_space.spaces['observation'].shape[0])

# These were pretty arbitrary models. We can use one of Mazes capabilities, the shape normalization #todo link
# with these models by wrapping them with the TorchModelBlock # todo: link to API reference
torch_model_block = TorchModelBlock(
    in_keys='observation', out_keys='action',
    in_shapes=observation_space.spaces['observation'].shape, in_num_dims=[2],
    out_num_dims=2, net=policy_net)

# Distribution Setup
# ------------------
# Initializing the proper probability distribution is rather easy with Maze. Simply provide the DistributionMapper
# with the action space and you automatically get the proper distribution to use. Optionally, you can specify
# a custom distribution. See # todo: link to distribution wrapper
distribution_mapper = DistributionMapper(action_space=action_space, distribution_mapper_config={})

"""
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
    policy=TorchPolicy(networks=policies,
                       distribution_mapper=distribution_mapper,
                       device=algorithm_config.device),
    critic=TorchSharedStateCritic(networks=critics,
                                  num_policies=1,
                                  device=algorithm_config.device),
    device=algorithm_config.device)

a2c = MultiStepA2C(env=envs, eval_env=eval_env, algorithm_config=algorithm_config, model=model,
                   model_selection=None)

setup_logging(job_config=None)

# train agent
a2c.train()

# final evaluation run
print("Final Evaluation Run:")
a2c.evaluate(deterministic=False, repeats=100)
"""