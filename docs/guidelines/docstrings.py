"""Provide a module docstring at the beginning of every Python file, describing why the file exists.

A good place to motivate and provide background to the data structure or algorithm in question, as opposed to
the more narrow class documentation
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from maze.core.annotations import override


class Interface(ABC):
    """One-line summary.

    Optional Paragraphs

    :param constructor_arg: Note that the __init__ arguments are documented in the class docstring.
    """

    def __init__(self, constructor_arg: str):
        # no docstring for __init__
        self.instance_variable = constructor_arg
        """Optionally document instance variables."""

        # provide type hints if the type can not be inferred from the initial assignment
        self.typed_instance_variable: Optional[str] = None

    @abstractmethod
    def interface_method(self, param1: List[int]) -> str:
        """One-line summary, not using variable names or the function name.

        All public functions and methods must provide docstrings with complete
        ``:param`` and ``:return`` blocks (except the function returns None)

        :param param1: Description of the parameter, mandatory.

        :return Description of the method return.
        """
        raise NotImplementedError()


class DerivedClass(Interface):
    """One-line summary.

    Optional description.
    """

    @override(Interface)  # override is mandatory
    def interface_method(self, param1: List[int]) -> str:
        """Do not copy the text from the parent class, instead describe what changed with respect to the parent method.

        The ``:param`` and/or ``:return:`` blocks can be skipped, if there is no new information to add to the existing
        parent method documentation.

        Other examples of valid docstrings:
        * "Implementation of ``Interface``."
          Bare minimum, in case there is absolutely nothing to be said about the
          implementation that is not already described in the super method.
        * "Forward call to :attr:`self.core_env <core_env>`."
          If the interface is implemented in some other method/class.
        """
