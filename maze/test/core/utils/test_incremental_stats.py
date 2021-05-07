"""Contains unit tests for incremental stats."""
import numpy as np
from maze.core.utils.stats_utils import CumulativeMovingMeanStd, CumulativeMovingMinMax


def test_incremental_mean_std():
    """ unit tests """

    # scalar case
    stats = CumulativeMovingMeanStd(epsilon=1e-8)
    data = []
    for i in range(5):
        stats.update(i)
        data.append(i)
        assert np.allclose(stats.mean, np.mean(data))
        assert np.allclose(stats.var, np.var(data))
    assert stats.mean.shape == ()
    assert stats.var.shape == ()

    # vector case
    stats = CumulativeMovingMeanStd(epsilon=1e-8)
    data = []
    for i in range(5):
        new_data = np.full(fill_value=i, shape=(5,))
        stats.update(new_data)
        data.append(new_data)
        assert np.allclose(stats.mean, np.mean(np.vstack(data)))
        assert np.allclose(stats.var, np.var(np.vstack(data)))
    assert stats.mean.shape == ()
    assert stats.var.shape == ()

    # matrix case
    stats = CumulativeMovingMeanStd(epsilon=1e-8)
    data = []
    for i in range(5):
        new_data = np.arange(start=i, stop=i+5)[np.newaxis]
        stats.update(new_data)
        data.append(new_data)
        assert np.allclose(stats.mean, np.mean(np.vstack(data), axis=0), atol=1e-6)
        assert np.allclose(stats.var, np.var(np.vstack(data), axis=0), atol=1e-6)
    assert stats.mean.shape == (5,)
    assert stats.var.shape == (5,)


def test_incremental_min_max():
    """ unit tests """

    # scalar case
    stats = CumulativeMovingMinMax(initial_min=5, initial_max=0)
    for i in range(5):
        stats.update(i)
        assert stats.min == 0
        assert stats.max == i
