"""
Tests observation normalization for environments.
"""
from functools import partial
from typing import Dict, Callable, Iterable

import gym
import numpy as np

from maze.core.env.maze_env import CoreEnvType
from maze.core.env.structured_env import StructuredEnv
from maze.core.utils.factory import ConfigType
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper


def estimate_normalization_statistics(env: ObservationNormalizationWrapper) -> ObservationNormalizationWrapper:
    """Estimates observation normalization statistics.
    :param env: A observation normalization wrapped structured environment.
    :return: The given env with initialized normalization parameters.
    """
    env.set_observation_collection(status=True)
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        env.step(action)
    env.estimate_statistics()
    env.set_observation_collection(status=False)

    return env


def conduct_observation_normalization_test(env: ObservationNormalizationWrapper,
                                           validation_callback: Callable,
                                           n_steps: int = 100) -> None:
    """
    Runs environment for n steps and validates collected observation statistics afterwards.
    Asserts that validation_callback returns True.
    :param env: Environment to run.
    :param validation_callback: Callable to check observation statistics. Callable is applied on every leaf in the
    observation space tree.
    :param n_steps: Number of steps to run.
    """
    act_conv_space: gym.spaces.space = env.action_conversion.space()

    for step in range(n_steps):
        observation = env.step(act_conv_space.sample())[0]
        for obs_key in observation:
            if obs_key not in env.exclude:
                assert validation_callback(observation[obs_key]), \
                    f"validation_callback not True for observation '{obs_key}'"


def match_observation_space_structure(space_a: Iterable, space_b: Iterable) -> bool:
    """
    Checks whether structure and shapes in space_a match those in space_b.
    :param space_a: First space.
    :param space_b: Second space.
    :return: True if shapes match, False if shape don't match or structure diverges.
    """

    # Check both spaces are of the same type.
    if not type(space_a) == type(space_b):
        return False

    # If observation space is a Dict: Examine sub-spaces recursively.
    if isinstance(space_a, dict) and isinstance(space_b, dict):
        if not space_a.keys() == space_b.keys():
            return False
        for key in space_a:
            if not match_observation_space_structure(space_a[key], space_b[key]):
                return False

    # If observation space is a Tuple: Update sub-spaces recursively.
    elif isinstance(space_a, tuple) and isinstance(space_b, tuple):
        if not len(space_a) == len(space_b):
            return False

        for subspace_a, subspace_b in zip(space_a, space_b):
            if not match_observation_space_structure(subspace_a, subspace_b):
                return False

    elif isinstance(space_a, np.ndarray) and isinstance(space_b, np.ndarray):
        if not space_a.shape == space_b.shape:
            return False

    # Note that we only consider three possible datatypes here: dict (equivalent to gym.spaces.Dict), tuple
    # (equivalent to gym.spaces.Tuple) and numerical numpy.ndarray (equivalent to gym.spaces.Box).
    # Other datatypes are only checked in terms of type equality.

    return True


def init_env_with_observation_normalization(env_factory: Callable[[], StructuredEnv], normalization_config: Dict) \
        -> ObservationNormalizationWrapper:
    """Instantiates new environment with normalized observations according to provided config.

    :param env_factory: A factory instantiating a structured environment.
    :param normalization_config: Observation normalization wrapper arguments (config) dictionary.
    :return: Observation normalization wrapped environment.
    """

    # initialize the env
    wrapped_env = ObservationNormalizationWrapper.wrap(env_factory(), **normalization_config)

    # estimate normalization statistics
    wrapped_env = estimate_normalization_statistics(wrapped_env)

    return wrapped_env


def env_registration_test(environment_type: type, wrapper_factory: Callable[[], ObservationNormalizationWrapper]) \
        -> None:
    """Tests initiation of an environment with normalization observation.

    :param environment_type: Environment type to assert for.
    :param wrapper_factory: A factory instantiating a observation normalized environment.
    """

    env: ObservationNormalizationWrapper = wrapper_factory()

    assert isinstance(env, environment_type)
    assert isinstance(env, ObservationNormalizationWrapper)

    env.close()


def observation_shape_match_test(env_factory: Callable[[], StructuredEnv],
                                 wrapper_factory: Callable[[], ObservationNormalizationWrapper]) -> None:
    """Tests if observation space shape(s) match between original observation space and normalized observation space.

    :param env_factory: A factory instantiating a structured environment.
    :param wrapper_factory: A factory instantiating a observation normalized environment.
    """

    env: CoreEnvType = env_factory()
    env_norm: ObservationNormalizationWrapper = wrapper_factory()

    assert match_observation_space_structure(
        env.step(env.action_conversion.space().sample())[0],
        env_norm.step(env.action_conversion.space().sample())[0]
    )


def range_zero_one_observation_value_range_test(wrapper_factory: Callable[[], ObservationNormalizationWrapper]) -> None:
    """
    Tests if normalized observation values lie within expected range for environment with
    RangeZeroOneNormalizedObservationConversion.

    :param wrapper_factory: A factory instantiating a observation normalized environment.
    """

    env: ObservationNormalizationWrapper = wrapper_factory()

    conduct_observation_normalization_test(
        env,
        lambda obs: np.all(0.0 <= obs) and np.all(obs <= 1.0),
        n_steps=10
    )

    env.close()


def run_observation_normalization_for_env(environment_type: type,
                                          env_factory: Callable[[], StructuredEnv],
                                          norm_config: ConfigType) -> None:
    """Runs observation normalization tests for maze environments.

    :param environment_type: Environment type to assert for.
    :param env_factory: A factory instantiating a structured environment.
    :param norm_config: Observation normalization wrapper arguments (config) dictionary.
    """

    # factory creating observation normalized environment
    wrapper_factory = partial(init_env_with_observation_normalization, env_factory, norm_config)

    # run individual tests
    env_registration_test(environment_type, wrapper_factory)
    observation_shape_match_test(env_factory, wrapper_factory)
    range_zero_one_observation_value_range_test(wrapper_factory)
