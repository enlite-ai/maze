""" Contains unit tests for maze core utility functions. """
from gym import spaces

from maze.core.utils.structured_env_utils import flat_structured_space, flat_structured_shapes, \
    stacked_shapes


def test_flat_structured_space_and_shapes():
    """ unit tests """
    structured_space_dict = dict()
    structured_space_dict[0] = spaces.Dict(spaces={"discrete": spaces.Discrete(5),
                                                   "binary": spaces.MultiBinary(5)})
    flat_space = flat_structured_space(structured_space_dict)
    assert isinstance(flat_space, spaces.Dict)
    assert "discrete" in flat_space.spaces
    assert "binary" in flat_space.spaces

    structured_space_dict[1] = structured_space_dict[0]
    flat_structured_space(structured_space_dict)

    structured_shape_dict = dict()
    structured_shape_dict[0] = {"discrete": (5,), "binary": (5,)}
    structured_shape_dict[1] = {"discrete": (5,), "binary": (5,)}
    flat_shapes = flat_structured_shapes(structured_shape_dict)
    assert isinstance(flat_shapes, dict)
    assert flat_shapes["discrete"] == (5,)
    assert flat_shapes["binary"] == (5,)


def test_stacked_shapes():
    agent_counts = {0: 1, 1: 3}
    shapes = {
        0: dict(a=(1,), b=(1, 2)),
        1: dict(c=(1,), d=(2, 2))
    }
    expected_stacked_shapes = {
        0: dict(a=(1,), b=(1, 2)),
        1: dict(c=(3, 1), d=(3, 2, 2))
    }

    assert expected_stacked_shapes == stacked_shapes(shapes, agent_counts)
