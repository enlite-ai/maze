"""Place for useful helpers to avoid duplicated code in the respective trainers."""
from collections import defaultdict
from typing import List, Dict, Iterable, Union

import numpy as np
import torch


def stack_numpy_dict_list(dict_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Stack list of dictionaries holding numpy arrays as values.

    :param dict_list: A list of identical dictionaries to be stacked, e.g. [{a: 1}, {a: 2}]
    :return: The list entries as a stacked dictionary, e.g. {a : [1, 2]}
    """
    list_dict = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            list_dict[k].append(v)

    stacked_dict = dict()
    for k in list_dict.keys():
        stacked_dict[k] = np.stack(list_dict[k])

    return stacked_dict


def unstack_numpy_list_dict(list_dict: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """Inverse of :func:`~maze.train.utils.train_utils.stack_numpy_dict_list`.

    Converts a dict of stacked lists (e.g. {a : [1, 2]}) into a list of dicts (e.g. [{a: 1}, {a: 2}]).

    :param list_dict: Dict of stacked lists, e.g. {a : [1, 2]}
    :return: List of dicts, e.g. [{a: 1}, {a: 2}]
    """
    keys = list(list_dict.keys())
    if list_dict[keys[0]].shape == ():
        # We already have flat values
        return [list_dict]

    n_items = len(list_dict[keys[0]])
    dict_list = []

    for i in range(n_items):
        action_dict = dict()
        for k in keys:
            action_dict[k] = list_dict[k][i]
        dict_list.append(action_dict)

    return dict_list


def compute_gradient_norm(params: Iterable[torch.Tensor]) -> float:
    """Computes the cumulative gradient norm of all provided parameters.

    :param params: Iterable over model parameters.
    :return: The cumulative gradient norm.
    """
    total_norm = 0.0
    for p in params:
        if p.requires_grad and np.prod(p.shape) > 0:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm


def stack_torch_dict_list(dict_list: List[Dict[str, Union[torch.Tensor, np.ndarray]]], dim: int = 0) -> Dict[str, torch.Tensor]:
    """Stack list of dictionaries holding torch tensors as values.

    Similar to :func:`~maze.train.utils.train_utils.stack_numpy_dict_list`, but for tensors.

    :param dict_list: A list of identical dictionaries to be stacked.
    :param dim: The dimension in which to stack/concat the lists.
    :return: The list entries as a stacked dictionary.
    """

    list_dict = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            list_dict[k].append(torch.from_numpy(v) if isinstance(v, np.ndarray) else v)

    stacked_dict = dict()
    for k in list_dict.keys():
        stacked_dict[k] = torch.stack(list_dict[k], dim=dim)

    return stacked_dict


def stack_torch_array_list(array_list: List[Union[np.ndarray, torch.Tensor]], expand: bool = False, dim: int = 0) \
        -> torch.Tensor:
    """Batch together a list of arrays (either torch or numpy) after converting them to torch. That is ether stack them
        if the batch dimension does not exists, otherwise concatenate them in the batch dimension (2)

    :param array_list: A list of arrays (either torch or numpy)
        :param expand: If True the values are expended by one dimension at dimension :param dim.
    :param dim: The dimension in which to stack/concat the lists.

    :return: the batched input
    """
    list_array = []
    for idx, array in enumerate(array_list):
        list_array.append(torch.from_numpy(array) if isinstance(array, np.ndarray) else array)

    if expand:
        stacked_list = torch.stack(list_array, dim=dim)
    else:
        stacked_list = torch.cat(list_array, dim=dim)

    return stacked_list
