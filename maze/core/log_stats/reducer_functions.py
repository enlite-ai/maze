"""Event log reducer functions."""
import itertools
from collections import ValuesView
from typing import List, Tuple, Union


def histogram(values: Union[List[Union[float, int]], List[List[Union[float, int]]],
                            List[ValuesView]]) -> Union[List[Union[float, int]], Tuple[List[Union[float, int]], int]]:
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

