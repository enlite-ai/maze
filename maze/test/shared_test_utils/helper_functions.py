""" Contains helper functions for unit testing """

import inspect
from typing import Tuple, Any, Dict, Type, List, Union

import numpy as np
from torch import nn

from maze.core.annotations import override
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_env import MazeEnv
from maze.perception.models.built_in.flatten_concat import FlattenConcatPolicyNet
from maze.perception.models.custom_model_composer import CustomModelComposer
from maze.perception.models.policies import ProbabilisticPolicyComposer
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_struct_env import DummyStructuredEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_structured_core_env import DummyStructuredCoreEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion


def build_dummy_base_env() -> BaseEnv:
    """ helper function creating a DummyBaseEnv for unit testing.
    :return: A Dummy Base Env.
    """

    class SomeBaseEnv(BaseEnv):
        """A dummy base env for unit testing.
        """

        @override(BaseEnv)
        def step(self, action: Any) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
            """ override of BaseEnv """
            return None, None, False, {}

        @override(BaseEnv)
        def reset(self) -> Any:
            """ override of BaseEnv """
            return None

        @override(BaseEnv)
        def seed(self, seed: int) -> None:
            """ override of BaseEnv """
            pass

        @override(BaseEnv)
        def close(self) -> None:
            """ override of BaseEnv """
            pass

    return SomeBaseEnv()


def build_dummy_maze_env() -> DummyEnvironment:
    """
    Instantiates the DummyEnvironment.

    :return: Instance of a DummyEnvironment
    """
    observation_conversion = ObservationConversion()

    return DummyEnvironment(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion]
    )


def build_dummy_structured_env() -> DummyStructuredEnvironment:
    """
    Instantiates a DummyStructuredEnvironment.

    :return: Instance of a DummyStructuredEnvironment
    """
    return DummyStructuredEnvironment(maze_env=build_dummy_maze_env())


def build_dummy_maze_env_with_structured_core_env() -> DummyEnvironment:
    """
    Instantiates the DummyEnvironment.

    :return: Instance of a DummyEnvironment
    """
    observation_conversion = ObservationConversion()

    return DummyEnvironment(
        core_env=DummyStructuredCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion]
    )


def all_classes_of_module(module) -> List[Type]:
    """Get all classes that are members of a module.

    :param module: the Python module
    :return a list of classes as Type objects
    """
    name_class_tuples = inspect.getmembers(module, inspect.isclass)
    return [t[1] for t in name_class_tuples]


def flatten_concat_probabilistic_policy_for_env(env: MazeEnv):
    """Build a probabilistic policy using a small flatten-concat network for a given env.

    Note: Supports structured envs with integer step keys.

    :param env: Env to build a policy for.
    """
    n_sub_steps = len(env.observation_spaces_dict.keys())

    composer = CustomModelComposer(
        action_spaces_dict=env.action_spaces_dict,
        observation_spaces_dict=env.observation_spaces_dict,
        agent_counts_dict=env.agent_counts_dict,
        distribution_mapper_config={},
        policy=dict(
            _target_=ProbabilisticPolicyComposer,
            networks=[dict(_target_=FlattenConcatPolicyNet, non_lin=nn.Tanh, hidden_units=[32, 32])] * n_sub_steps,
            substeps_with_separate_agent_nets=[]
        ),
        critic=None
    )

    return composer.policy


def convert_np_array_to_tuple(arr: np.ndarray) -> Union[Tuple, np.ndarray]:
    """
    Recursive conversion of numpy arrays with an arbitrary number of dimensions to tuples.
    :param arr: numpy array to convert.
    :return: n-dimensional tuple.
    """

    try:
        return tuple(convert_np_array_to_tuple(i) for i in arr)
    except TypeError:
        return arr
