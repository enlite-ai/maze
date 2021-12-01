"""Event log reducer functions."""
import itertools
from collections import ValuesView
from typing import List, Tuple, Union

import numpy as np


def epoch_quantile_25(values: List[List[Union[int, float]]]) -> float:
    """Computes the 25th quantile on epoch level assuming that no reduction took place on episode level.

    :param values: List of episode value lists.
    :return: The 25th quantile.
    """
    all_epoch_values = np.concatenate(values)
    return np.quantile(all_epoch_values, q=0.25)


def epoch_quantile_75(values: List[List[Union[int, float]]]) -> float:
    """Computes the 75th quantile on epoch level assuming that no reduction took place on episode level.

    :param values: List of episode value lists.
    :return: The 75th quantile.
    """
    all_epoch_values = np.concatenate(values)
    return np.quantile(all_epoch_values, q=0.75)


def epoch_median(values: List[List[Union[int, float]]]) -> float:
    """Computes the median on epoch level assuming that no reduction took place on episode level.

    :param values: List of episode value lists.
    :return: The median.
    """
    all_epoch_values = np.concatenate(values)
    return float(np.median(all_epoch_values))


def epoch_mean(values: List[List[Union[int, float]]]) -> float:
    """Computes the mean on epoch level assuming that no reduction took place on episode level.

    :param values: List of episode value lists.
    :return: The mean.
    """
    all_epoch_values = np.concatenate(values)
    return float(np.mean(all_epoch_values))


def histogram(values: Union[List[Union[float, int]], List[List[Union[float, int]]], List[ValuesView]]) \
        -> Union[List[Union[float, int]], Tuple[List[Union[float, int]], int]]:
    """the histogram reducer function
        We decided to return the full list, rather then binning the values (e.g. collections.Counter), so that float
        values are supported as well.

    :param values: A list of values collected by the event system
    :return: returns the same list of values so that a histogram can then be build from it
    """
    assert isinstance(values, list), 'Only lists supported so far'

    # For multiple parallel environments, values[0] can be of type list (we get a list of actions for each of the
    # environments). These multiple lists have to be flattened.
    if len(values) > 0 and isinstance(values[0], list):
        return list(itertools.chain.from_iterable(values))

    return values
