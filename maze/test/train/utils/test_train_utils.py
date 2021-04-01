import numpy as np
import torch

from maze.perception.perception_utils import convert_to_torch
from maze.train.utils.train_utils import unstack_numpy_list_dict, stack_numpy_dict_list, stack_torch_dict_list


def stacked_example():
    """Dictionary of stacked arrays, i.e. stacked form. Corresponds to unstacked_example."""
    return {
        "a": np.array([[1], [2]]),
        "b": np.array([[3], [4]])
    }


def unstacked_example():
    """List of dictionaries with flat arrays, i.e. unstacked form. Corresponds to stacked_example."""
    return [
        {"a": np.array([1]), "b": np.array([3])},
        {"a": np.array([2]), "b": np.array([4])},
    ]


def test_numpy_list_dict_unstacking():
    assert unstack_numpy_list_dict(stacked_example()) == unstacked_example()


def test_numpy_dict_list_stacking():
    stacked = stack_numpy_dict_list(unstacked_example())
    for k, v in stacked_example().items():
        assert np.all(v == stacked[k])


def test_torch_conversion():
    stacked = convert_to_torch(stack_numpy_dict_list(unstacked_example()), in_place=True, cast=None, device="cpu")
    for k, v in convert_to_torch(stacked_example(), in_place=True, cast=None, device="cpu").items():
        assert torch.all(v == stacked[k])


def test_torch_dict_list_stacking():
    unstacked_ex = convert_to_torch(unstacked_example(), in_place=True, cast=None, device="cpu")
    stacked_ex = convert_to_torch(stacked_example(), in_place=True, cast=None, device="cpu")

    stacked = stack_torch_dict_list(unstacked_ex)
    for k, v in stacked_ex.items():
        assert torch.all(v == stacked[k])
