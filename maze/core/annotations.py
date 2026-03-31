"""Provides commonly used annotation decorators."""

from __future__ import annotations


def override(cls: type):
    """
    Annotation for documenting method overrides.

    :param cls: The superclass that provides the overridden method. If this
                cls does not actually have the method, an error is raised.
    """

    def _check_override(method):
        if getattr(cls, '__sphinx_mock__', None) is None:
            if method.__name__ not in dir(cls):
                raise NameError(f'{method} does not override any method of {cls}')

        fully_qualified_name = '.'.join([str(cls.__module__), str(cls.__name__)])
        method.__doc__ = f'(overrides :class:`~{fully_qualified_name}`)\n\n{method.__doc__}'

        return method

    return _check_override


# noinspection PyUnusedLocal
def unused(*args):
    """Function to annotate unused variables. Also disables the 'unused parameter/value' inspection warning."""
