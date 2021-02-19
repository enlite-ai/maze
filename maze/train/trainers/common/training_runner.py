"""Defines the base classes und general utility functions for training runners."""
import os
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Any, Optional

from omegaconf import DictConfig

from maze.core.utils.config_utils import EnvFactory, SwitchWorkingDirectoryToInput
from maze.core.utils.registry import Registry
from maze.core.wrappers.observation_normalization.observation_normalization_utils import \
    obtain_normalization_statistics, make_normalized_env_factory
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.models.space_config import SpacesConfig
from maze.runner import Runner
from maze.train.trainers.common.trainer import Trainer
from maze.utils.bcolors import BColors
from maze.utils.log_stats_utils import setup_logging


@dataclass
class AlgorithmConfig(ABC):
    """Base class for all specific algorithm configurations."""


@dataclass
class ModelConfig:
    """Model configuration structure.

    As with TrainConfig this class enables type hinting, but is not actually instantiated."""

    policies: Dict[Any, Any]
    critics: Optional[Dict[Any, Any]]
    distribution_mapper: Dict[Any, Any]


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


@dataclass
class TrainingRunner(Runner):
    """Base class for training runner implementations."""

    state_dict_dump_file: str
    """Where to save the best model (output directory handled by hydra)"""

    spaces_config_dump_file: str
    """Where to save the env spaces configuration (output directory handled by hydra)"""

    normalization_samples: int
    """Number of samples (=steps) to collect normalization statistics at the beginning of the training."""

    def __init__(self, state_dict_dump_file: str, spaces_config_dump_file: str, normalization_samples: int):
        self.state_dict_dump_file = state_dict_dump_file
        self.spaces_config_dump_file = spaces_config_dump_file
        self.normalization_samples = normalization_samples

        self.env_factory = None
        self.model_composer = None
        self.normalization_statistics = None

    def run(self, cfg: DictConfig) -> None:
        """
        While this method is designed to be overriden by individual subclasses, it provides some functionality
        that is useful in general:

        - Building the env factory for env + wrappers
        - Estimating normalization statistics from the env
        - If successfully estimated, wrapping the env factory so that envs are already built with the statistics
        - Building the model composer from model config and env spaces config
        - Serializing the env spaces configuration (so that the model composer can be re-loaded for future rollout)
        - Initializing logging setup

        :param cfg: Full Hydra run job config
        """
        with SwitchWorkingDirectoryToInput(cfg.input_dir):
            # if the observation normalization is already available, read it from the input directory
            self.env_factory = EnvFactory(cfg.env, cfg.wrappers if "wrappers" in cfg else {})
            normalization_env = self.env_factory()

        # Observation normalization
        self.normalization_statistics = obtain_normalization_statistics(normalization_env,
                                                                        n_samples=self.normalization_samples)
        if self.normalization_statistics:
            self.env_factory = make_normalized_env_factory(self.env_factory, self.normalization_statistics)
            # dump statistics to current working directory
            assert isinstance(normalization_env, ObservationNormalizationWrapper)
            normalization_env.dump_statistics()

        # init model composer
        composer_type = Registry(base_type=BaseModelComposer).class_type_from_module_name(cfg.model['type'])
        composer_type.check_model_config(cfg.model)
        self.model_composer = Registry(base_type=BaseModelComposer).arg_to_obj(
            cfg.model,
            action_spaces_dict=normalization_env.action_spaces_dict,
            observation_spaces_dict=normalization_env.observation_spaces_dict)

        SpacesConfig(self.model_composer.action_spaces_dict,
                     self.model_composer.observation_spaces_dict).save(self.spaces_config_dump_file)

        # Should be done after the normalization runs, otherwise stats from those will get logged as well.
        setup_logging(job_config=cfg)

    @classmethod
    def _init_trainer_from_input_dir(cls, trainer: Trainer, state_dict_dump_file: str, input_dir: str) -> None:
        """Initialize trainer from given state dict and input directory.
        :param trainer: The trainer to initialize.
        :param state_dict_dump_file: The state dict dump file relative to input_dir.
        :param input_dir: The directory to load the state dict from.
        """
        with SwitchWorkingDirectoryToInput(input_dir):
            if os.path.exists(state_dict_dump_file):
                BColors.print_colored(f"Model initialized from '{state_dict_dump_file}' of '{input_dir}'!",
                                      BColors.OKGREEN)
                trainer.load_state(state_dict_dump_file)
            else:
                BColors.print_colored("Model initialized with random parameters! ", BColors.OKGREEN)
