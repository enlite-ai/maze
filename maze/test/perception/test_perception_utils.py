"""Test the perception utils"""
import numpy as np
import pytest
import torch
from gym import spaces

from maze.perception.perception_utils import flatten_spaces, observation_spaces_to_in_shapes, \
    map_nested_structure, convert_to_torch, convert_to_numpy, stack_and_flatten_spaces


def test_flat_structured_observations():
    """Test the two perception utils methods"""
    obs_space1 = spaces.Dict({'obs1': spaces.Box(low=-1, high=1, shape=(20,))})
    obs_space2 = spaces.Dict({'obs1': spaces.Box(low=-1, high=1, shape=(20,)),
                              'obs3': spaces.Box(low=-9999, high=9999, shape=(2,))})
    obs_space = {0: obs_space1, 1: obs_space2}
    structured_obs = {kk: {k: v.sample() for k, v in vv.spaces.items()} for kk, vv in obs_space.items()}
    structured_obs[0]['obs1'] = structured_obs[1]['obs1']

    _ = flatten_spaces(structured_obs.values())
    _ = observation_spaces_to_in_shapes(obs_space)


def test_stack_and_flatten_spaces():
    observations = [dict(a=torch.ones(3, 4), b=torch.ones(3, 2)), dict(b=torch.ones(3, 2), c=torch.ones(3))]
    expected_shapes = dict(a=(3, 4), b=(2, 3, 2), c=(3,))

    for space_name, space_value in stack_and_flatten_spaces(spaces=observations, dim=0).items():
        assert space_value.shape == expected_shapes[space_name]


def test_map_nested_structure_mutable():
    # Single elem
    test_value = torch.rand((20, 4))
    out_value = map_nested_structure(test_value, mapping=lambda x: x, in_place=True)
    assert test_value is out_value

    test_value = torch.rand((20, 4))
    out_value = map_nested_structure(test_value, mapping=lambda x: x, in_place=False)
    assert test_value is not out_value

    # List
    test_value = [torch.rand((20, 4))]
    out_value = map_nested_structure(test_value, mapping=lambda x: x, in_place=True)
    assert test_value is out_value

    test_value = [torch.rand((20, 4))]
    out_value = map_nested_structure(test_value, mapping=lambda x: x, in_place=False)
    assert test_value is not out_value

    test_value = [torch.rand((2, 4))]
    out_value = map_nested_structure(test_value, mapping=lambda x: x, in_place=False)
    test_value[0] *= 2
    assert test_value is not out_value


def test_map_nested_structure_inmutable():
    test_value = tuple(torch.rand((2, 4)))
    with pytest.raises(Exception) as e_info:
        out_value = map_nested_structure(test_value, mapping=lambda x: x * 2, in_place=True)

    test_value = list(torch.rand((2, 4)))
    out_value = map_nested_structure(test_value, mapping=lambda x: x * 2, in_place=True)
    assert test_value is out_value


def test_convert_to_torch():
    if torch.cuda.is_available():
        test_value = [torch.randn((2, 4))]
        out_value = convert_to_torch(test_value, device='cuda', cast=None, in_place=True)
        assert test_value is out_value

    test_value = [torch.randn((2, 4))]
    out_value = convert_to_torch(test_value, device=None, cast=torch.float64, in_place=True)
    assert test_value is out_value

    if torch.cuda.is_available():
        test_value = [torch.randn((2, 4))]
        out_value = convert_to_torch(test_value, device='cuda', cast=None, in_place='try')
        assert test_value is out_value

    if torch.cuda.is_available():
        test_value = torch.randn((2, 4))
        with pytest.raises(Exception) as e_info:
            out_value = convert_to_torch(test_value, device='cuda', cast=None, in_place=True)

    if torch.cuda.is_available():
        test_value = torch.randn((2, 4))
        out_value = convert_to_torch(test_value, device='cuda', cast=None, in_place='try')
        assert test_value is not out_value

    test_value = torch.randn((2, 4))
    with pytest.raises(Exception) as e_info:
        out_value = convert_to_torch(test_value, device=None, cast=torch.float64, in_place=True)


def test_base_types():
    test_value = list([np.random.random((4, 2))])
    out_value = map_nested_structure(test_value, mapping=lambda x: torch.from_numpy(x), in_place=True)

    def foo(elem):
        elem[0] = elem[0] * 2

    foo(test_value)
    assert test_value is out_value
    assert test_value[0] is out_value[0]
    assert isinstance(out_value[0], torch.Tensor)
    assert isinstance(test_value[0], torch.Tensor)

    # test in_place false
    test_value = list([np.random.random((4, 2))])
    out_value = map_nested_structure(test_value, mapping=lambda x: torch.from_numpy(x), in_place=False)

    foo(test_value)
    assert test_value is not out_value
    assert test_value[0] is not out_value[0]
    assert isinstance(out_value[0], torch.Tensor)
    assert isinstance(test_value[0], np.ndarray)


def test_convert_to_numpy():
    """ unit tests """
    stats = dict()
    stats["key_1"] = torch.from_numpy(np.random.random(10))
    convert_to_numpy(stats=stats, cast=None, in_place=False)
    convert_to_numpy(stats=stats, cast=None, in_place=True)
    convert_to_numpy(stats=stats, cast=None, in_place='try')
    assert isinstance(stats["key_1"], np.ndarray)
