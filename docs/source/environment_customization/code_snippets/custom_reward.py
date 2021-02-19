"""Assigns negative reward for relying on raw pieces for delivering an order."""
from typing import List

from maze.core.annotations import override
from maze.core.events.pubsub import Subscriber
from maze_envs.logistics.cutting_2d.env.events import InventoryEvents
from maze.core.env.reward import RewardAggregatorInterface


class RawPieceUsageRewardAggregator(RewardAggregatorInterface):
    """
    Reward scheme for the 2D cutting env penalizing raw piece usage.

    :param reward_scale: Reward scaling factor.
    """
    def __init__(self, reward_scale: float):
        super().__init__()
        self.reward_scale = reward_scale

    @override(Subscriber)
    def get_interfaces(self) -> List:
        """Specification of the event interfaces this subscriber wants to receive events from.
        Every subscriber must implement this configuration method.

        :return: A list of interface classes.
        """
        return [InventoryEvents]

    def summarize_reward(self) -> float:
        """Summarize reward based on the orders and pieces to cut.

        :return: the summarized scalar reward.
        """

        # iterate replenishment events and assign reward accordingly
        reward = 0.0
        for _ in self.query_events(InventoryEvents.piece_replenished):
            reward -= 1.0

        # rescale reward with provided factor
        reward *= self.reward_scale

        return reward

    @classmethod
    @override(RewardAggregatorInterface)
    def to_scalar_reward(cls, reward: float) -> float:
        """Nothing to do here for this env as the reward is already a scalar.

        This method is useful for example in a multi-agent setting
        where we could sum over multiple actors to assign a joint reward.

        :param reward: Here already a scalar reward.
        :return: The scalar reward returned by the environment.
        """
        return reward
