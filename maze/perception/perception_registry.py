""" Registry for the perception module """

from maze.core.utils.registry import Registry
from maze.perception import blocks as blocks_module
from maze.perception.blocks.base import PerceptionBlock


class PerceptionRegistry:
    """The Perception Module's Registry.

    blocks: The dynamically and manually registered perception blocks.
    """

    blocks = Registry(root_module=blocks_module, base_type=PerceptionBlock)
