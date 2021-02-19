"""
Getting started example from:
https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html
"""

import gym
from stable_baselines3 import A2C

# ENV INSTANTIATION
# -----------------
env = gym.make('CartPole-v0')

# TRAINING AND ROLLOUT
# --------------------

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
