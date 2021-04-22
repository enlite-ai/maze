"""Contains utility functions for structured environments."""
from typing import Dict, Union, Sequence

from gym import spaces

from maze.core.env.structured_env import StepKeyType


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


def flat_structured_shapes(shapes: Dict[StepKeyType, Dict[str, Sequence[int]]]) -> Dict[str, Sequence[int]]:
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


def stacked_shapes(shapes: Dict[StepKeyType, Dict[str, Sequence[int]]],
                   agent_counts_dict: Dict[StepKeyType, int]) -> Dict[StepKeyType, Dict[str, Sequence[int]]]:
    """Adopt shapes dict for stacked multi-agent scenario. (Mainly for observation stacking.)

    Takes a dict of shapes for a structured step. Augments shapes for sub-steps that have multiple agents
    by the agent dimension, so the observations from multiple agents can be concatenated.
    """
    shapes = shapes.copy()

    for sub_step_key, agent_count in agent_counts_dict.items():
        # Dynamic number of agents is not supported
        assert agent_count != -1, "for observation concatenation, the number of agents " \
                                  "needs to be known upfront"

        # Single-agent spaces are left as-is
        if agent_count == 1:
            continue

        # Multi-agent spaces are stacked
        for obs_name, original_shape in shapes[sub_step_key].items():
            shapes[sub_step_key][obs_name] = (agent_count, *original_shape)

    return shapes
