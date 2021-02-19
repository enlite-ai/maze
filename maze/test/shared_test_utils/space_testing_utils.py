"""Contains gym space helper functions. """

import gym
import numpy as np


def build_dict_obs_space():
    """build dictionary observation space"""
    space_dict = dict()
    space_dict["features"] = gym.spaces.Box(low=0.0, high=1.0, shape=(24,), dtype=np.float64)
    space_dict["stacked_features"] = gym.spaces.Box(low=0.0, high=1.0, shape=(100, 24, 2), dtype=np.float64)
    space_dict["feature_series"] = gym.spaces.Box(low=0.0, high=1.0, shape=(64, 24), dtype=np.float64)
    space_dict["categorical_feature"] = gym.spaces.Box(low=0, high=11, shape=(), dtype=np.int)
    space_dict["categorical_feature_series"] = gym.spaces.Box(low=0, high=11, shape=(64,), dtype=np.int)
    space_dict["image"] = gym.spaces.Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float64)
    space_dict["image_series"] = gym.spaces.Box(low=0.0, high=1.0, shape=(64, 3, 32, 32), dtype=np.float64)
    space_dict["image_rgb"] = gym.spaces.Box(low=0.0, high=1.0, shape=(32, 32, 3), dtype=np.float64)

    return gym.spaces.Dict(space_dict)