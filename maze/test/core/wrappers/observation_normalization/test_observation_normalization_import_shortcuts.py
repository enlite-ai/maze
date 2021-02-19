""" Contains tests for the observation normalization wrapper. """
from maze import normalization_strategies
from maze.core.utils.registry import Registry
from maze.core.wrappers.observation_normalization import normalization_strategies as normalization_strategies_module
from maze.core.wrappers.observation_normalization.normalization_strategies.base import ObservationNormalizationStrategy


def test_normalization_strategy_import_shortcuts():
    """Tests if all strategies have shortcuts added to the normalization_strategies/__init__.py"""

    # get list of all registered maze preprocessors
    registry = Registry(base_type=ObservationNormalizationStrategy,
                        root_module=normalization_strategies_module)

    # iterate preprocessors
    for strategy in list(registry.__dict__["type_registry"].values()):
        assert hasattr(normalization_strategies, strategy.__name__)
