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

    def clone_from(self, reward_aggregator: 'RewardAggregatorInterface') -> None:
        """Clones the state of the provided reward aggregator.

        :param reward_aggregator: The reward aggregator to clone from.
        """
        for key, value in self.__dict__.items():
            if key != "events":
                assert value == reward_aggregator.__dict__[key], \
                    f"Your reward aggregator seems to be stateful. Make sure to overwrite 'clone_from' properly!"
