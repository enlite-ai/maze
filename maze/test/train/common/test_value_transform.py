"""Contains unit tests for training utility functions."""
import numpy as np
import torch
from maze.train.trainers.common.value_transform import support_to_scalar, scalar_to_support, ReduceScaleValueTransform


def test_transform_value():
    """ unit test """
    x = np.arange(-100, 100, 0.1)
    eps = 1e-7
    transform = ReduceScaleValueTransform(epsilon=eps)
    x_hat = transform.transform_value_inv(transform.transform_value(x))
    assert np.allclose(x, x_hat)


def test_support_to_scalar():
    """ unit test """

    # single vector
    logits = np.full(11, fill_value=0, dtype=np.float32)
    logits[3] = 0.3
    logits[4] = 0.7
    logits = torch.from_numpy(logits)
    scalar = support_to_scalar(logits, support_range=(0, 10))
    assert scalar.shape == ()
    assert np.allclose(scalar, 4.861410140991211)

    # batch of vectors
    logits = np.full((10, 11), fill_value=0, dtype=np.float32)
    logits[:, 3] = 0.3
    logits[:, 4] = 0.7
    logits = torch.from_numpy(logits)
    scalar = support_to_scalar(logits, support_range=(0, 10)).cpu().numpy()
    assert scalar.shape == (10,)
    assert np.allclose(scalar, 4.861410140991211)


def test_scalar_to_support():
    """ unit test """
    scalar = torch.scalar_tensor(3.7)

    # single scalar value
    support = scalar_to_support(scalar=scalar, support_range=(0, 10))
    support = support.cpu().numpy()
    assert support.shape == (11,)
    assert np.allclose(support[3], 0.3)
    assert np.allclose(support[4], 0.7)

    support = scalar_to_support(scalar=scalar, support_range=(-10, 10))
    support = support.cpu().numpy()
    assert support.shape == (21,)
    assert np.allclose(support[13], 0.3)
    assert np.allclose(support[14], 0.7)

    # vector of scalars
    scalar = torch.from_numpy(np.full(shape=(10,), fill_value=3.7, dtype=np.float32))
    support = scalar_to_support(scalar=scalar, support_range=(-10, 10))
    assert support.shape == (10, 21)
    assert np.allclose(support[:, 13], 0.3)
    assert np.allclose(support[:, 14], 0.7)
