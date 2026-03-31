"""Contains unit tests for examples."""

from __future__ import annotations

from examples.es_gym_cartpole import main


def test_es_gym_cartpole():
    """unit tests"""
    main(n_epochs=1)
