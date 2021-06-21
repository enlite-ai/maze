"""Interface specifying the computation of scalar rewards from aggregated reward events."""
from abc import abstractmethod, ABC
from typing import Union, List, Type, Optional

import numpy as np
from maze.core.annotations import override
from maze.core.env.maze_state import MazeStateType
from maze.core.events.pubsub import Subscriber


class RewardAggregatorInterface(Subscriber):
    """Event aggregation object for reward customization and shaping."""

    @abstractmethod
    def summarize_reward(self, maze_state: Optional[MazeStateType] = None) -> Union[float, np.ndarray]:
        """
        Summarize the reward for this step. Expected to be called once per structured step.

        Calculates the reward based either on the current maze state of the environment
        (provided as an argument), or by querying the events dispatched by the environment
        during the last step. The former is simpler, the latter is more flexible, as
        environment state does not always carry all the necessary information on what took place in the last step.

        In most scenarios, the reward is returned either as a scalar, or as an numpy array corresponding
        to the number of actors that acted in the last step. Scalar is useful in scenarios with single actor
        only, or when per-actor reward cannot be easily attributed and a shared scalar reward makes more sense.
        An array is useful in scenarios where per-actor reward makes sense, such as in multi-agent setting.

        :param maze_state: Current state of the environment.
        :return: Reward for the last structured step. In most cases, either a scalar or an array with an item
                 for each actor active during the last step.
        """

    @override(Subscriber)
    def get_interfaces(self) -> List[Type[ABC]]:
        """
        Declare which events this reward aggregator should be notified about.

        Often, the current state of the environment does not provide enough information to calculate a reward. In such
        cases, the reward aggregator collects events from the environment. (E.g., was a new piece replenished during
        cutting? Did the agent attempt an invalid cut?) This method declares which events this aggregator
        should collect.

        By default, this returns an empty list (as for simpler cases, maze state is enough and no events are needed).

        For more complex scenarios, override this method and specify which interfaces are needed.

        :return: A list of event interface classes to listen to
        """
        return []

    def clone_from(self, reward_aggregator: 'RewardAggregatorInterface') -> None:
        """Clones the state of the provided reward aggregator.

        :param reward_aggregator: The reward aggregator to clone from.
        """
        for key, value in self.__dict__.items():
            if key != "events":
                assert value == reward_aggregator.__dict__[key], \
                    f"Your reward aggregator seems to be stateful. Make sure to overwrite 'clone_from' properly!"
