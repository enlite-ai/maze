"""Contains an example showing how to use observation normalization directly from python."""
from maze.core.agent.random_policy import RandomPolicy
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper
from maze.core.wrappers.observation_normalization.observation_normalization_utils import \
    obtain_normalization_statistics

# instantiate a maze environment
env = GymMazeEnv("CartPole-v1")

# this is the normalization config as a python dict
normalization_config = {
    "default_strategy": "maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
    "default_strategy_config": {"clip_range": (None, None), "axis": 0},
    "default_statistics": None,
    "statistics_dump": "statistics.pkl",
    "sampling_policy": RandomPolicy(env.action_spaces_dict),
    "exclude": None,
    "manual_config": None
}

# 1. PREPARATION: first we estimate normalization statistics
# ----------------------------------------------------------

# wrap the environment for observation normalization
env = ObservationNormalizationWrapper.wrap(env, **normalization_config)

# before we can start working with normalized observations
# we need to estimate the normalization statistics
normalization_statistics = obtain_normalization_statistics(env, n_samples=1000)

# 2. APPLICATION (training, rollout, deployment)
# ----------------------------------------------

# instantiate a maze environment
training_env = GymMazeEnv("CartPole-v1")
# wrap the environment for observation normalization
training_env = ObservationNormalizationWrapper.wrap(training_env, **normalization_config)

# reuse the estimated the statistics in our training environment(s)
training_env.set_normalization_statistics(normalization_statistics)

# after this step the training env yields normalized observations
normalized_obs = training_env.reset()
