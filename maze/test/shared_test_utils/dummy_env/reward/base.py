"""
An empty reward aggregator which just passes the reward through
"""

from abc import ABC
from typing import List, Type

from maze.core.env.reward import RewardAggregatorInterface


class RewardAggregator(RewardAggregatorInterface):
    """Event aggregation object dealing with cutting rewards.
    """

    def get_interfaces(self) -> List[Type[ABC]]:
        """
        A emtpy get_interfaces function
        """
        pass

    def summarize_reward(self) -> float:
        """Summarize reward based on the orders and pieces to cut.

        :return: the summarized scalar reward.
        """
        raise NotImplementedError

    @classmethod
    def to_scalar_reward(cls, reward: float) -> float:
        """Nothing to do here for this env.

        :param: reward: already a scalar reward
        :return: the same scalar reward
        """
        return reward
