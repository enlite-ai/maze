""" Contains tests for the perception module. """
import maze
from maze.core.utils.registry import Registry
from maze.perception.blocks.base import PerceptionBlock
from maze.perception.blocks import feed_forward, recurrent, joint_blocks, general
from maze.perception.builders.base import BaseModelBuilder
from maze.perception import builders
from maze.perception.models import critics
from maze.perception.models import policies

from maze.perception.models.critics import BaseStateCriticComposer
from maze.perception.models.policies.base_policy_composer import BasePolicyComposer


def test_blocks_import_shortcuts():
    """Tests if all blocks have shortcuts in blocks/__init__.py"""

    for module in [feed_forward, recurrent, joint_blocks, general]:
        registry = Registry(base_type=PerceptionBlock, root_module=module)

        # iterate preprocessors
        for block in list(registry.__dict__["type_registry"].values()):
            assert hasattr(maze.perception.blocks, block.__name__)


def test_builders_import_shortcuts():
    """Tests if all builders have shortcuts in blocks/__init__.py"""
    registry = Registry(base_type=BaseModelBuilder, root_module=builders)

    # iterate preprocessors
    for block in list(registry.__dict__["type_registry"].values()):
        assert hasattr(maze.perception.builders, block.__name__)


def test_critics_composers_import_shortcuts():
    """Tests if all critic composers have shortcuts in critic/__init__.py"""
    registry = Registry(base_type=BaseStateCriticComposer, root_module=critics)

    # iterate preprocessors
    for critic in list(registry.__dict__["type_registry"].values()):
        print(critic.__name__)
        assert hasattr(maze.perception.models.critics, critic.__name__)


def test_policy_composers_import_shortcuts():
    """Tests if all policy composers have shortcuts in policies/__init__.py"""
    registry = Registry(base_type=BasePolicyComposer, root_module=policies)

    # iterate preprocessors
    for policy in list(registry.__dict__["type_registry"].values()):
        print(policy.__name__)
        assert hasattr(maze.perception.models.policies, policy.__name__)
