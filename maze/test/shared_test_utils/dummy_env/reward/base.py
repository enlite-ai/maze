"""
An empty reward aggregator which just passes the reward through
"""

from abc import ABC
from typing import List, Type

from maze.core.env.reward import RewardAggregatorInterface


class DummyEnvEvents(ABC):
    """Minimal event class for the DummyCoreEnv"""
    def twice_per_step(self, value: int):
        """A dummy event that is called twice per step."""


class RewardAggregator(RewardAggregatorInterface):
    """Event aggregation object dealing with cutting rewards.
    """

    def get_interfaces(self) -> List[Type[ABC]]:
        """
        A emtpy get_interfaces function
        """
        return [DummyEnvEvents]

    def summarize_reward(self) -> float:
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
