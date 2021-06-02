""" Contains utility functions for the perception module. """
import collections
import copy
from collections import defaultdict
from typing import Union, Dict, Any, Callable, Sequence, Iterable

import gym
import numpy as np
import torch


def observation_spaces_to_in_shapes(observation_spaces: Dict[Union[int, str], gym.spaces.Dict]) \
        -> Dict[Union[int, str], Dict[str, Sequence[int]]]:
    """Convert an observation space to the input shapes for the neural networks

    :param observation_spaces: the observation spaces of a structured Env
    :return: the same structure but all the gym spaces are converted to tuples
    """
    in_shapes = dict()
    for obs_key, obs_dict in observation_spaces.items():
        in_shapes[obs_key] = dict()
        assert isinstance(obs_dict, gym.spaces.Dict)
        for key, value in obs_dict.spaces.items():
            assert isinstance(value, gym.spaces.Box), 'Only box observation spaces supported at this point, but got: ' \
                                                      f'{type(value)}'
            in_shapes[obs_key][key] = value.shape

    return in_shapes


def flatten_spaces(spaces: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Merges an iterable of dictionary spaces (usually observations or actions from subsequent sub-steps)
    into a single dictionary containing all the items.

    If one key is present in multiple elements, its value will be present only once in the resulting dictionary,
    and all values for such key will be checked for a match.

    :param: Iterable of dictionary spaces (usually observations or actions from subsequent sub-steps).
    :return: One flat dictionary, containing all keys and values form the elements of the iterable.
    """
    result = dict()

    for space in spaces:
        for key, obs in space.items():
            # check if the heads match if strict mode
            if key in result:
                assert result[key].shape == obs.shape
                if isinstance(obs, np.ndarray):
                    assert np.allclose(result[key], obs), f"Tried merging observations, but values for key {key} " \
                                                          "differ. Did you intend to stack the observations instead?"
                else:
                    assert torch.allclose(result[key], obs), f"Tried merging observations, but values for key {key} " \
                                                             "differ. Did you intend to stack the observations instead?"

            # override with newest value if already existing
            result[key] = obs

    return result


def stack_and_flatten_spaces(spaces: Iterable[Dict[str, torch.Tensor]], dim: int) -> Dict[str, torch.Tensor]:
    """Merges an iterable of dictionary spaces (usually observations or actions from subsequent sub-steps)
    into a single dictionary containing all the items.

    If one key is present in multiple elements, all its values will be concatenated in the resulting dictionary.

    :param spaces: Iterable of dictionary spaces (usually observations or actions from subsequent sub-steps).
    :param dim: Dimension along which to stack (usually 0 if we have a single environment, or 1 if we have a batch
                of environments)
    :return: One flat dictionary, containing all keys and values form the elements of the iterable.
    """
    result = defaultdict(list)

    # Collect observations in flat dict
    for space in spaces:
        for key, obs in space.items():
            result[key].append(obs)

    # Concatenate all at once for efficiency
    for obs_name, observations in result.items():
        if len(observations) == 1:
            result[obs_name] = observations[0]
        else:
            result[obs_name] = torch.stack(observations, dim=dim)

    # Return an ordinary dict, not default dict
    return dict(result)


def convert_to_torch(stats: Any, device: Union[str, None], cast: Union[torch.dtype, None],
                     in_place: Union[bool, str]):
    """Converts any struct to torch.Tensors.

    :param stats: Any (possibly nested) struct, the values in which will be
        converted and returned as a new struct with all leaves converted
        to torch tensors.
    :param device: 'cpu' or 'cuda', or None if it should stay the same
    :param cast: the type the element should be cast to, or None if it should stay the same
    :param in_place: specify if the operation should be done in_place, can be bool or 'try'

    :return: A new struct with the same structure as `stats`, but with all
            values converted to torch Tensor types.

    """

    def mapping(item):
        """Define mapping"""
        if torch.is_tensor(item):
            tensor = item
        else:
            tensor = torch.from_numpy(np.asarray(item))
        if tensor.dtype != cast and cast is not None:
            tensor = tensor.to(cast)
        if tensor.device != device and device is not None:
            tensor = tensor.to(device)
        return tensor

    assert in_place in [True, False, 'try']
    if in_place == 'try':
        return map_nested_structure(stats, mapping, in_place, 1)
    else:
        return map_nested_structure(stats, mapping, in_place)


def convert_to_numpy(stats: Any, cast: Union[np.dtype, None], in_place: Union[bool, str]):
    """Convert torch to np

    :param stats: Any (possibly nested) struct, the values in which will be
        converted and returned as a new struct with all leaves converted
        to torch tensors.
    :param cast: if the element should also be casted to a specific type
    :param in_place: specify if the operation should be done in_palce, can be bool or 'try'

    :return: A new struct with the same structure as `stats`, but with all
            values converted to torch Tensor types. can be bool or 'try'
    """

    def mapping(item):
        """Define mapping"""
        if isinstance(item, torch.Tensor):
            item = item.cpu().numpy()
        # Cast the tensor to the desired format
        return item if cast is None else item.astype(cast)

    assert in_place in [True, False, 'try']
    if in_place == 'try':
        return map_nested_structure(stats, mapping, in_place, 1)
    else:
        return map_nested_structure(stats, mapping, in_place)


def map_nested_structure(nested_instance: Any,
                         mapping: Callable[[Union[torch.Tensor, np.ndarray, int, float]],
                                           Union[torch.Tensor, np.ndarray, int, float]], in_place: bool,
                         _depth: int = 0) -> Any:
    """Apply a custom callable to an nested object where the base elements are either torch.Tensor, np.ndarray,
        int or float.

        :param nested_instance: the nested instance that should be mapped.
        :param mapping: the mapping that should be applied to the base instance.
        :param in_place: specifies if the mapping should be done in_place (mutating the given object), if this is set
            to true but not possible (with immutable objects) an exception is thrown.
        :param _depth: a counter for the recursion depth
        """
    # List, iterators, generators
    if isinstance(nested_instance, torch.Tensor) or isinstance(nested_instance, np.ndarray) \
            or isinstance(nested_instance, int) or isinstance(nested_instance, float):
        # If it is a torch.Tensor or np.ndarray object
        if in_place:
            out = mapping(nested_instance)
            if _depth == 0 and nested_instance is not out:
                raise Exception('The given nested structure could not be mapped in-place (as specified) since '
                                'the method had to be applied at depth 0, and the mapped results is not the same')
            return out
        else:
            if isinstance(nested_instance, torch.Tensor):
                return mapping(nested_instance.clone().detach())
            elif isinstance(nested_instance, np.ndarray):
                return mapping(np.copy(nested_instance))
            else:
                return mapping(nested_instance)
    else:
        if isinstance(nested_instance, dict):
            if in_place:
                for key, value in nested_instance.items():
                    nested_instance[key] = map_nested_structure(value, mapping, in_place, _depth + 1)
                return nested_instance
            else:
                new_nested_instance = dict()
                for key, value in nested_instance.items():
                    new_nested_instance[key] = map_nested_structure(value, mapping, in_place, _depth + 1)
                return new_nested_instance

        elif isinstance(nested_instance, Sequence):
            if isinstance(nested_instance, collections.MutableSequence):
                if in_place:
                    for idx, value in enumerate(nested_instance):
                        nested_instance[idx] = map_nested_structure(value, mapping, in_place, _depth + 1)
                    return nested_instance
                else:
                    new_nested_instance = type(nested_instance)()
                    for idx, value in enumerate(nested_instance):
                        new_nested_instance.insert(idx, map_nested_structure(value, mapping, in_place, _depth + 1))
                    return new_nested_instance

            elif isinstance(nested_instance, tuple):
                if in_place:
                    raise Exception('The given nested structure could not be mapped in-place (as specified) since '
                                    'a tuple was encountered. Please revisit the function call')
                else:
                    new_nested_instance = type(nested_instance)()
                    for idx, value in enumerate(nested_instance):
                        new_nested_instance += (map_nested_structure(value, mapping, in_place, _depth + 1),)
                    return new_nested_instance

            else:
                raise Exception('Not supported Sequence substructure type found: {}'.format(type(nested_instance)))

        else:
            raise Exception('Not supported Structure type found: {}'.format(type(nested_instance)))
