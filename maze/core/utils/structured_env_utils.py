"""Contains utility functions for structured environments."""
from typing import Dict, Union, Tuple, Sequence

from gym import spaces


def flat_structured_space(structured_space_dict: Dict[Union[int, str], spaces.Dict]) -> spaces.Dict:
    """Compiles a flat gym.spaces.Dict space from a structured environment space.
    :param: The structured dictionary spaces.
    :return: The flattened dictionary space.
    """
    flat_dict = dict()
    for sub_step_key, sub_step_space in structured_space_dict.items():
        assert isinstance(sub_step_space, spaces.Dict)

        for key, space in sub_step_space.spaces.items():
            # check if the heads match
            if key in flat_dict:
                assert flat_dict[key] == space, f"Key '{key}' already contained in flat space dictionary!"

            flat_dict[key] = space
    return spaces.Dict(spaces=flat_dict)


def flat_structured_shapes(shapes: Dict[Union[str, int], Dict[str, Sequence[int]]]) -> Dict[str, Sequence[int]]:
    """Flatten a dict of shape dicts to a single dict

    :param shapes: Collection of shape dict.
    :return: Flat shape dict.
    """
    result = dict()

    for sub_step_key, sub_step_space in shapes.items():
        for key, action_space in sub_step_space.items():
            # check if the heads match
            if key in result:
                assert result[key] == action_space

            result[key] = action_space

    return result
