"""
Auxiliary routines for tests related to observation statistics.
"""

import base64
import hashlib
from typing import Tuple, Any, List, Callable, Union, Iterable

import gym
import numpy as np

from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper


def append_observation_to_statistics(observation: Iterable, statistics: dict):
    """
    Stores observation.
    :param: Observation to append
    :param: Statistics dict in which to store the observation.
    :return: Updated statistics.
    """

    def process_key(_key: Any):
        """
        Processes observation for current observation.
        :param _key:
        """
        if _key in statistics:
            if type(observation[_key]) in (np.ndarray, list):
                if "values" not in statistics[_key]:
                    statistics[_key]["values"] = []
                statistics[_key]["values"].append(observation[_key])
            else:
                append_observation_to_statistics(observation[_key], statistics[_key])

    if isinstance(observation, dict):
        for key in observation:
            process_key(key)
    elif isinstance(observation, tuple):
        for i, _ in enumerate(observation):
            process_key(i)


def validate_observation_statistics(statistics: dict, validation_callback: Callable):
    """
    Computes statistics for gathered observations.
    :param statistics: Gathered observations.
    :param validation_callback: Callback for statistical measures to perform on every leaf in the observation space
    tree. Structure: measure name -> Callable.
    """

    def process_key(_key: Any):
        """
        Processes current observation subspace.
        :param _key: Key to current observation subspace.
        """
        # Check whether we are in the last leaf.
        if "min" in statistics[_key] and isinstance(statistics[_key]["min"], np.ndarray):
            # Compute statistics.
            statistics[_key]["min"] = np.min(np.asarray(statistics[_key]["values"]))
            statistics[_key]["max"] = np.max(np.asarray(statistics[_key]["values"]))
            statistics[_key]["interval"] = statistics[_key]["max"] - statistics[_key]["min"]
            statistics[_key].pop("values")

            # Execute validation callbacks.
            assert validation_callback(statistics[_key]), "Validation callback failed."
        else:
            validate_observation_statistics(statistics[_key], validation_callback)

    if isinstance(statistics, dict):
        for key in statistics:
            process_key(key)
    elif isinstance(statistics, tuple):
        for i, _ in enumerate(statistics):
            process_key(i)


def conduct_observation_statistics_validation_test(
    env: ObservationNormalizationWrapper, validation_callback: Callable, n_steps: int = 100
):
    """
    Runs environment for n steps and validates collected observation statistics afterwards.
    Asserts that validation_callback returns True.
    :param env: Environment to run.
    :param validation_callback: Callable to check observation statistics. Callable is applied on every leaf in the
    observation space tree.
    :param n_steps: Number of steps to run.
    """

    act_conv_space: gym.spaces.space = env.action_conversion.space()
    stats: dict = env.fetch_statistics()

    for step in range(n_steps):
        append_observation_to_statistics(env.step(act_conv_space.sample())[0], stats)

    # Perform check to ensure values are not too far outside statisticsal bounds.
    # Value range to be checked is to be discussed, delta of +- 0.1 was arbitrarily chosen.
    validate_observation_statistics(stats, validation_callback)
