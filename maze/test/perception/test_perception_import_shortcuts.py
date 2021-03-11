""" Contains tests for the perception module. """

import pytest

import maze
from maze.perception import builders
from maze.perception.blocks import feed_forward, recurrent, joint_blocks, general
from maze.perception.models import critics
from maze.perception.models import policies
from maze.test.shared_test_utils.helper_functions import all_classes_of_module


@pytest.mark.parametrize("module", [feed_forward, recurrent, joint_blocks, general])
def test_blocks_import_shortcuts(module):
    """Tests if all blocks have shortcuts in blocks/__init__.py"""

    # iterate preprocessors
    for block in all_classes_of_module(module):
        assert hasattr(maze.perception.blocks, block.__name__)


def test_builders_import_shortcuts():
    """Tests if all builders have shortcuts in blocks/__init__.py"""
    # iterate preprocessors
    for block in all_classes_of_module(builders):
        assert hasattr(maze.perception.builders, block.__name__)


def test_critics_composers_import_shortcuts():
    """Tests if all critic composers have shortcuts in critic/__init__.py"""
    # iterate preprocessors
    for critic in all_classes_of_module(critics):
        print(critic.__name__)
        assert hasattr(maze.perception.models.critics, critic.__name__)


def test_policy_composers_import_shortcuts():
    """Tests if all policy composers have shortcuts in policies/__init__.py"""
    # iterate preprocessors
    for policy in all_classes_of_module(policies):
        print(policy.__name__)
        assert hasattr(maze.perception.models.policies, policy.__name__)
