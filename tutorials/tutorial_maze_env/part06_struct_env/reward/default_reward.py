from typing import List

from maze.core.annotations import override
from maze.core.env.reward import RewardAggregatorInterface
from ..env.events import CuttingEvents, InventoryEvents


class DefaultRewardAggregator(RewardAggregatorInterface):
    """Default reward scheme for the 2D cutting env.

    :param invalid_action_penalty: Negative reward assigned for an invalid cutting specification.
    :param raw_piece_usage_penalty: Negative reward assigned for starting a new raw inventory piece.
    """

    def __init__(self, invalid_action_penalty: float, raw_piece_usage_penalty: float):
        super().__init__()
        self.invalid_action_penalty = invalid_action_penalty
        self.raw_piece_usage_penalty = raw_piece_usage_penalty

    @override(RewardAggregatorInterface)
    def get_interfaces(self):
        """Specification of the event interfaces this subscriber wants to receive events from.
        Every subscriber must implement this configuration method.
        :return: A list of interface classes"""
        return [CuttingEvents, InventoryEvents]

    @override(RewardAggregatorInterface)
    def summarize_reward(self) -> List[float]:
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

        return sum(rewards)
