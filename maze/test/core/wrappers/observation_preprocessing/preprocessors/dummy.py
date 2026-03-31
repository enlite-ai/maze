"""Contains a dummy pre-processor for unit testing."""

from __future__ import annotations

from maze.core.annotations import override
from maze.core.wrappers.observation_preprocessing.preprocessors.flatten import FlattenPreProcessor


class DummyPreProcessor(FlattenPreProcessor):
    """Dummy pre-processor for unit testing.
    (Just the same as the FlattenPreProcessor)
    """

    @override(FlattenPreProcessor)
    def tag(self) -> str:
        """implementation of
        :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.flatten.FlattenPreProcessor` interface
        """
        return __name__.rsplit('.')[-1]
