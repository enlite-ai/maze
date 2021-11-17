""" Contains a simple replay buffer"""
from typing import Union, List

import numpy as np

from maze.core.annotations import override
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord
from maze.train.trainers.common.replay_buffer.replay_buffer import BaseReplayBuffer


class UniformReplayBuffer(BaseReplayBuffer):
    """ Replay buffer for off policy learning.

    :param buffer_size: The maximum buffer size.
    :param seed: The random seed used for initializing the uniform random sampling in the buffer.
    """

    def __init__(self, buffer_size: int, seed: int):
        self._buffer_size = int(buffer_size)
        self._buffer = np.full(shape=(buffer_size,), fill_value=np.nan, dtype=np.object)
        self._buffer_idx = 0
        self._fill_state = 0
        self.cum_moving_avg_num_picks = 0.0
        self._total_samples_count = 0
        self._total_number_of_transitions = 0
        self.buffer_rng = np.random.RandomState(seed)

    @override(BaseReplayBuffer)
    def add_transition(self, transition: Union[StructuredSpacesRecord, SpacesTrajectoryRecord]) -> None:
        """implementation of :class:`~maze.train.trainers.common.replay_buffer.replay_buffer.BaseReplayBuffer`
        """
        self._buffer[self._buffer_idx] = transition
        self._fill_state = max(self._buffer_idx + 1, self._fill_state)
        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size
        self._total_number_of_transitions += 1

    @override(BaseReplayBuffer)
    def add_rollout(self, rollout: Union[SpacesTrajectoryRecord, List[StructuredSpacesRecord]]) -> None:
        """implementation of :class:`~maze.train.trainers.common.replay_buffer.replay_buffer.BaseReplayBuffer`
        """
        if isinstance(rollout, SpacesTrajectoryRecord):
            step_records = rollout.step_records
        elif isinstance(rollout, list):
            step_records = rollout
        else:
            raise ValueError(f'Not supported input type: {type(rollout)}')

        for ii in step_records:
            self.add_transition(ii)

    @override(BaseReplayBuffer)
    def sample_batch(self, n_samples: int, learner_device: str) -> \
            List[Union[StructuredSpacesRecord, SpacesTrajectoryRecord]]:
        """implementation of :class:`~maze.train.trainers.common.replay_buffer.replay_buffer.BaseReplayBuffer`
        """
        indices = self.buffer_rng.permutation(len(self))[:n_samples]
        sample_batch = self._buffer[indices]
        self._total_samples_count += len(indices)
        self.cum_moving_avg_num_picks = float(self._total_samples_count) / float(self._total_number_of_transitions)
        return sample_batch

    @override(BaseReplayBuffer)
    def __len__(self) -> int:
        """implementation of :class:`~maze.train.trainers.common.replay_buffer.replay_buffer.BaseReplayBuffer`
        """
        return self._fill_state
