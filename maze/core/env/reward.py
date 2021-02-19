"""Interface specifying the computation of scalar rewards from aggregated reward events."""
from abc import abstractmethod
from typing import Any

from maze.core.events.pubsub import Subscriber


class RewardAggregatorInterface(Subscriber):
    """Event aggregation object for reward customization and shaping.
    """

    @classmethod
    @abstractmethod
    def to_scalar_reward(cls, reward: Any) -> float:
        """Aggregate sub-rewards to scalar reward.

        This method is useful for example in a multi-agent setting
        where we could sum over multiple actors to assign a joint reward.

        :param: reward: The aggregated reward (e.g. per-agent reward for multi-agent RL settings).
        :return: The scalar reward returned by the environment.
        """
