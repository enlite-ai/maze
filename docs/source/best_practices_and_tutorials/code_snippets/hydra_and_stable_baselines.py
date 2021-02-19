"""
Contains an example showing how to train an observation normalized maze environment
instantiated from a hydra config with stable-baselines.
"""

from maze.core.utils.config_utils import make_env_from_hydra
from maze.core.wrappers.no_dict_spaces_wrapper import NoDictSpacesWrapper
from maze.core.wrappers.observation_normalization.observation_normalization_utils import \
    obtain_normalization_statistics

from stable_baselines3 import A2C

# ENV INSTANTIATION: from hydra config file
# -----------------------------------------
env = make_env_from_hydra("conf")

# OBSERVATION NORMALIZATION
# -------------------------

# next we estimate the normalization statistics by
# (1) collecting observations by randomly sampling 1000 transitions from the environment
# (2) computing the statistics according to the define normalization strategy
normalization_statistics = obtain_normalization_statistics(env, n_samples=1000)
env.set_normalization_statistics(normalization_statistics)

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
