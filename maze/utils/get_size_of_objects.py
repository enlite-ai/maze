"""File containing method for calculating the size of a given python object"""

import sys
from gc import get_referents
from types import ModuleType, FunctionType
from typing import Any, Tuple

BLACKLIST = type, ModuleType, FunctionType


def getsize(object: Any) -> Tuple[int, int]:
    """Calculate the size of a given python object.

    :param object: The object the size of which should be calculated.
    :return: A tuple holding the size in byte and gbyte
    """
    if isinstance(object, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: ' + str(type(object)))
    seen_ids = set()
    size_in_byte = 0
    objects = [object]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size_in_byte += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)

    size_in_gbyte = size_in_byte * 9.31 * (10 ** -10)
    return size_in_byte, size_in_gbyte
