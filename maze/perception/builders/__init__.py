""" Import blocks to enable import shortcuts. """
from maze.perception.builders.base import BaseModelBuilder
from maze.perception.builders.concat import ConcatModelBuilder

assert issubclass(ConcatModelBuilder, BaseModelBuilder)
