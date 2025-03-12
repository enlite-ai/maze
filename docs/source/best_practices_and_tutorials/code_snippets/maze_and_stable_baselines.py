"""
Contains an example showing how to train
an observation normalized maze environment with stable-baselines.
"""

from maze.core.agent.random_policy import RandomPolicy
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.core.wrappers.no_dict_spaces_wrapper import NoDictSpacesWrapper
from maze.core.wrappers.observation_normalization.observation_normalization_utils import \
    obtain_normalization_statistics
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper

from stable_baselines3 import A2C

# ENV INSTANTIATION: a GymMazeEnv instead of a gymnasium.Env
# ----------------------------------------------------
env = GymMazeEnv('CartPole-v1')

# OBSERVATION NORMALIZATION
# -------------------------

# we wrap the environment with the ObservationNormalizationWrapper
# (you can find details on this in the section on observation normalization)
env = ObservationNormalizationWrapper(
    env=env,
    default_strategy="maze.normalization_strategies.MeanZeroStdOneObservationNormalizationStrategy",
    default_strategy_config={"clip_range": (None, None), "axis": 0},
    default_statistics=None, statistics_dump="statistics.pkl",
    sampling_policy=RandomPolicy(env.action_spaces_dict),
    exclude=None, manual_config=None)

# next we estimate the normalization statistics by
# (1) collecting observations by randomly sampling 1000 transitions from the environment
# (2) computing the statistics according to the define normalization strategy
normalization_statistics = obtain_normalization_statistics(env, n_samples=1000)
env.set_normalization_statistics(normalization_statistics)

# after this step all observations returned by the environment will be normalized

# stable-baselines does not support dict spaces so we have to remove them
env = NoDictSpacesWrapper(env)

# TRAINING AND ROLLOUT (remains unchanged)
# ----------------------------------------

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
