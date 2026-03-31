"""Contains unit tests for examples."""

from __future__ import annotations

from examples.a2c_gym_cartpole import main


def test_a2c_gym_cartpole():
    """unit tests"""
    main(n_epochs=1)
