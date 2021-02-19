"""
Tests for wrapper configuration in .yaml config file.
"""

from typing import TypeVar

import pytest

import maze
import maze.test.core.wrappers.dummy_wrappers as dummy_wrappers_module
import maze.test.core.wrappers.dummy_wrappers_for_duplication_test as dummy_wrappers_for_duplication_test_module
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_env import MazeEnv
from maze.core.utils.registry import Registry
from maze.core.wrappers.wrapper import Wrapper
from maze.core.wrappers.wrapper_registry import WrapperRegistry
from maze.test.core.wrappers.dummy_wrappers.dummy_wrapper_a import DummyWrapperA
from maze.test.core.wrappers.dummy_wrappers.dummy_wrapper_b import DummyWrapperB
from maze.test.core.wrappers.dummy_wrappers.dummy_wrappers import DummyWrapper
from maze.test.shared_test_utils.config_testing_utils import load_env_config
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion

T = TypeVar('T')


def test_instantiation_with_wrapper_registry():
    """
    Test instantiation, types and parsing for/of a wrapped (VR) environment.
    """

    # Register dummy wrappers.
    registry = WrapperRegistry([
        maze.core.wrappers,
        maze.test.core.wrappers.dummy_wrappers.dummy_wrapper_a,
        maze.test.core.wrappers.dummy_wrappers.dummy_wrapper_b])

    default_config: dict = load_env_config(dummy_wrappers_module, "dummy_env_config_with_dummy_wrappers.yml")
    env_config: dict = default_config['env']
    env_config["core_env"] = {"observation_space": ObservationConversion().space()}
    env: Wrapper[MazeEnv] = registry.wrap_from_config(
        Registry.build_obj(DummyEnvironment, env_config),
        default_config['wrappers']
    )

    # Make sure types are correctly inferred.
    assert isinstance(env, Wrapper)
    assert isinstance(env, DummyWrapper)
    assert isinstance(env, DummyWrapperA)
    assert isinstance(env, DummyWrapperB)
    assert isinstance(env, DummyEnvironment)

    # Check if arguments are set correctly and methods are available.
    assert getattr(env, "do_stuff")
    assert getattr(env, "arg_a")
    assert getattr(env, "arg_b")
    assert getattr(env, "arg_c")
    assert env.do_stuff() == "b"


def test_wrap_method():
    """
    Tests .wrap() method.
    """

    default_config: dict = load_env_config(dummy_wrappers_module, "dummy_env_config_with_dummy_wrappers.yml")
    env_config: dict = default_config['env']
    env_config["core_env"] = {"observation_space": ObservationConversion().space()}
    env: DummyEnvironment = DummyEnvironment(**env_config)

    env_a: DummyWrapperA = DummyWrapperA.wrap(env, arg_a=1)
    assert isinstance(env_a, DummyWrapperA)

    try:
        DummyWrapperB.wrap(env)
        raise Exception("Wrapping shouldn't work without specifying the needed arguments.")
    except TypeError:
        pass

    env_b: DummyWrapperB = DummyWrapperB.wrap(env, arg_b=2, arg_c=3)
    assert isinstance(env_b, DummyWrapperB)


def test_wrapper_registration():
    """
    Tests dynamic registration of wrapper classes.
    """

    # # Test registration of wrappers in tests.
    # reg_wrappers: WrapperRegistry = WrapperRegistry([maze.test.core.wrappers.dummy_wrappers])
    # assert "ObservationNormalizationWrapper" not in reg_wrappers
    # assert "TimeLimitWrapper" not in reg_wrappers
    # assert "LogStatsWrapper" not in reg_wrappers
    # assert "DummyWrapperA" in reg_wrappers
    # assert "DummyWrapperB" in reg_wrappers
    # assert "Wrapper" not in reg_wrappers
    # assert "VehicleRoutingCoreEnvironment" not in reg_wrappers
    #
    # # Test registration of wrappers in core.
    # reg_wrappers: WrapperRegistry = WrapperRegistry(
    #     [maze.core.wrappers]
    # )
    #
    # assert "ObservationNormalizationWrapper" in reg_wrappers
    # assert "TimeLimitWrapper" in reg_wrappers
    # assert "LogStatsWrapper" in reg_wrappers
    # assert "DummyWrapperA" not in reg_wrappers
    # assert "DummyWrapperB" not in reg_wrappers
    # assert "Wrapper" not in reg_wrappers
    # assert "VehicleRoutingCoreEnvironment" not in reg_wrappers
    #
    # # Test merging of registration from multiple paths.
    # reg_wrappers: WrapperRegistry = WrapperRegistry(
    #     [maze.core.wrappers, dummy_wrappers_module]
    # )
    #
    # assert "ObservationNormalizationWrapper" in reg_wrappers
    # assert "TimeLimitWrapper" in reg_wrappers
    # assert "LogStatsWrapper" in reg_wrappers
    # assert "DummyWrapperA" in reg_wrappers
    # assert "DummyWrapperB" in reg_wrappers
    # assert "Wrapper" not in reg_wrappers
    # assert "VehicleRoutingCoreEnvironment" not in reg_wrappers
    #
    # reg_wrappers: WrapperRegistry = WrapperRegistry(
    #     [maze.core.wrappers, dummy_wrappers_module]
    # )
    #
    # assert "ObservationNormalizationWrapper" in reg_wrappers
    # assert "TimeLimitWrapper" in reg_wrappers
    # assert "LogStatsWrapper" in reg_wrappers
    # assert "DummyWrapperA" in reg_wrappers
    # assert "DummyWrapperB" in reg_wrappers
    # assert "Wrapper" not in reg_wrappers
    # assert "VehicleRoutingCoreEnvironment" not in reg_wrappers

    # Make sure we can't register an additional module with the same name as an existing, registered one.
    with pytest.raises(AssertionError):
        WrapperRegistry(
            [dummy_wrappers_module,
             dummy_wrappers_for_duplication_test_module]
        )
