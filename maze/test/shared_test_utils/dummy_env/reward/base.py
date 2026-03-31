"""
An empty reward aggregator which just passes the reward through
"""

from __future__ import annotations

from abc import ABC

from maze.core.env.maze_state import MazeStateType
from maze.core.env.reward import RewardAggregatorInterface

# ruff: noqa: B027, B024


class DummyEnvEvents(ABC):
    """Minimal event class for the DummyCoreEnv"""

    def twice_per_step(self, value: int):
        """A dummy event that is called twice per step."""


class RewardAggregator(RewardAggregatorInterface):
    """Event aggregation object dealing with cutting rewards."""

    def get_interfaces(self) -> list[type[ABC]]:
        """
        A empty get_interfaces function
        """
        return [DummyEnvEvents]

    def summarize_reward(self, maze_state: MazeStateType | None = None) -> float:  # noqa: ARG002
        """Summarize reward based on the orders and pieces to cut.

        :return: the summarized scalar reward.
        """
        return sum(e.value for e in self.query_events(DummyEnvEvents.twice_per_step))

    @classmethod
    def to_scalar_reward(cls, reward: float) -> float:
        """Nothing to do here for this env.

        :param: reward: already a scalar reward
        :return: the same scalar reward
        """
        return reward
