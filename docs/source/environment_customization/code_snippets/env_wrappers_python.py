"""Contains an example showing how to add wrappers."""
from maze.core.wrappers.random_reset_wrapper import RandomResetWrapper
from maze.core.wrappers.time_limit_wrapper import TimeLimitWrapper

# instantiate the environment
env = ...

# apply wrappers
env = RandomResetWrapper.wrap(env, min_skip_steps=0, max_skip_steps=100)
env = TimeLimitWrapper.wrap(env, max_episode_steps=1000)
