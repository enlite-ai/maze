""" Contains tests for the observation normalization wrapper. """
from maze import normalization_strategies
from maze.core.wrappers.observation_normalization import normalization_strategies as normalization_strategies_module
from maze.test.shared_test_utils.helper_functions import all_classes_of_module


def test_normalization_strategy_import_shortcuts():
    """Tests if all strategies have shortcuts added to the normalization_strategies/__init__.py"""

    # iterate preprocessors
    for strategy in all_classes_of_module(normalization_strategies_module):
        assert hasattr(normalization_strategies, strategy.__name__)
