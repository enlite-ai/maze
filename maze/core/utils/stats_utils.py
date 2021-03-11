"""Contains statistics helper utils."""
from typing import Optional, Union

import numpy as np


class CumulativeMovingMeanStd(object):
    """Maintains cumulative moving mean and std of incoming numpy arrays along axis 0.

    Output shapes:
    scalar -> scalar
    vector -> scalar
    matrix -> vector

    Implementation adopted from:
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py

    :param epsilon: Ensures numerical stability and avoids division by zero.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.mean: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None
        self._count: float = epsilon

    def update(self, new_data: Union[np.ndarray, float]) -> None:
        """Update cumulative moving statistics.

        :param new_data: New data to update the stats with.
        """
        if not isinstance(new_data, np.ndarray):
            new_data = np.asarray([new_data])

        batch_mean = new_data.mean(axis=0)
        batch_var = new_data.var(axis=0)

        if self.mean is None:
            self.mean = np.zeros_like(batch_mean)
            self.var = np.zeros_like(batch_var)

        batch_count = new_data.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """Update cumulative moving statistics.

        :param batch_mean: Mean of new incoming data.
        :param batch_var: Variance of new incoming data.
        :param batch_count: Samples contained in new incoming data.
        """
        mean_diff = batch_mean - self.mean
        new_count = self._count + batch_count

        # cumulative update of mean
        new_mean = self.mean + mean_diff * batch_count / new_count
        # cumulative update of variance
        m_2 = (self.var * self._count) + (batch_var * batch_count)
        m_2 += np.square(mean_diff) * self._count * batch_count / new_count
        new_var = m_2 / new_count

        # update statistics
        self.mean = new_mean
        self.var = new_var
        self._count = new_count
