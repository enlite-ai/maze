""" Contains the base replay buffer interface."""
from abc import abstractmethod
from typing import Union, List

import numpy as np

from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord


class BaseReplayBuffer:
    """Abstract interface for all replay buffer implementations."""

    @abstractmethod
    def add_rollout(self, rollout: Union[SpacesTrajectoryRecord, List[StructuredSpacesRecord]]) -> None:
        """Add an actor rollout to the buffer.

        :param rollout: A single actor rollout consisting of n_rollout_steps transitions.
        """

    @abstractmethod
    def sample_batch(self, n_samples: int, learner_device: str) -> List[Union[StructuredSpacesRecord,
                                                                              SpacesTrajectoryRecord]]:
        """Sample mini-batch randomly from the buffer.

        :param n_samples: The number of samples to draw.
        :param learner_device: The device of the learner (cpu or cuda).
        :return: A sample batch of trajectory or spaces records
        """

    @abstractmethod
    def add_transition(self, transition: Union[StructuredSpacesRecord, SpacesTrajectoryRecord]) -> None:
        """Add a single transition (rollout length == 1) to the buffer.

        :param transition: The actor transition to be added to the buffer.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Retrieve the current fill state of the buffer.

        :return: Return the current fill state of the buffer.
        """
