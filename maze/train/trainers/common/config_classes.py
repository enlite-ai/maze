"""
Classes reflecting component configuration.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any

from maze.runner import Runner

from omegaconf import DictConfig


@dataclass
class AlgorithmConfig(ABC):
    """Base class for all specific algorithm configurations."""

    n_epochs: int
    """number of epochs to train"""


@dataclass
class ModelConfig:
    """Model configuration structure.

    As with TrainConfig this class enables type hinting, but is not actually instantiated."""

    policies: dict[Any, Any]
    critics: dict[Any, Any] | None
    distribution_mapper: dict[Any, Any]


@dataclass
class TrainConfig:
    """Top-level configuration structure.

    The structured configuration support of hydra is limited currently (1.0-rc2).

    E.g.

    - Merging different configuration files did not work as expected  (e.g. algorithm and env-algorithm)
    - Although the entry-point expects a TrainConfig object, it just receives a DictConfig, which can cause
      unexpected behaviour.

    Note that due to this limitations, this class merely acts as type hinting mechanism. Behind the scenes
    we receive raw DictConfig objects and either need to invoke the ``Registry`` functionality or
    ``hydra.utils.instantiate`` to instantiated objects of specific types where required.
    """

    env: DictConfig
    model: ModelConfig
    algorithm: AlgorithmConfig
    runner: Runner
