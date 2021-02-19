from abc import abstractmethod
from typing import List

from maze.core.env.reward import RewardAggregatorInterface

from ..env.events import CuttingEvents, InventoryEvents


class CuttingRewardAggregator(RewardAggregatorInterface):
    """Interface for cutting reward aggregators."""

    @abstractmethod
    def collect_rewards(self) -> List[float]:
        """Assign rewards and penalties according to respective events.
        :return: List of individual event rewards.
        """


class DefaultRewardAggregator(CuttingRewardAggregator):
    """Default reward scheme for the 2D cutting env.

    :param invalid_action_penalty: Negative reward assigned for an invalid cutting specification.
    :param raw_piece_usage_penalty: Negative reward assigned for starting a new raw inventory piece.
    """

    def __init__(self, invalid_action_penalty: float, raw_piece_usage_penalty: float):
        super().__init__()
        self.invalid_action_penalty = invalid_action_penalty
        self.raw_piece_usage_penalty = raw_piece_usage_penalty

    def get_interfaces(self):
        """Specification of the event interfaces this subscriber wants to receive events from.
        Every subscriber must implement this configuration method.
        :return: A list of interface classes"""
        return [CuttingEvents, InventoryEvents]

    def collect_rewards(self) -> List[float]:
        """Assign rewards and penalties according to respective events.
        :return: List of individual event rewards.
        """

        rewards: List[float] = []

        # penalty for starting a new raw inventory piece
        for _ in self.query_events(InventoryEvents.piece_replenished):
            rewards.append(self.raw_piece_usage_penalty)

        # penalty for selecting an invalid piece for cutting
        for _ in self.query_events(CuttingEvents.invalid_piece_selected):
            rewards.append(self.invalid_action_penalty)

        # penalty for specifying invalid cutting parameters
        for _ in self.query_events(CuttingEvents.invalid_cut):
            rewards.append(self.invalid_action_penalty)

        return rewards

    @classmethod
    def to_scalar_reward(cls, reward: List[float]) -> float:
        """Aggregate sub-rewards to scalar reward.

        This method is useful for example in a multi-agent setting
        where we could sum over multiple actors to assign a joint reward.

        :param: reward: The aggregated reward (e.g. per-agent reward for multi-agent RL settings).
        :return: The scalar reward returned by the environment.
        """
        return sum(reward)
