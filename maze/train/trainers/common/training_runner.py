"""Defines the base classes und general utility functions for training runners."""

import logging
import os
import dataclasses
from typing import Optional, Callable, Union

from omegaconf import DictConfig

from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.utils.config_utils import EnvFactory, SwitchWorkingDirectoryToInput
from maze.core.utils.factory import Factory
from maze.core.wrappers.observation_normalization.normalization_strategies.base import StructuredStatisticsType
from maze.core.utils.seeding import set_seeds_globally, MazeSeeding
from maze.core.wrappers.observation_normalization.observation_normalization_utils import \
    obtain_normalization_statistics, make_normalized_env_factory
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper
from maze.core.wrappers.wrapper_factory import WrapperFactory
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.models.space_config import SpacesConfig
from maze.runner import Runner
from maze.train.trainers.common.model_selection.best_model_selection import BestModelSelection
from maze.train.trainers.common.trainer import Trainer
from maze.utils.bcolors import BColors
from maze.utils.log_stats_utils import setup_logging


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainingRunner(Runner):
    """
    Base class for training runner implementations.
    """

    state_dict_dump_file: str
    """Where to save the best model (output directory handled by hydra)."""
    spaces_config_dump_file: str
    """Where to save the env spaces configuration (output directory handled by hydra)."""
    normalization_samples: int
    """Number of samples (=steps) to collect normalization statistics at the beginning of the
    training."""

    env_factory: Optional[
        Union[EnvFactory, Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, ObservationNormalizationWrapper]]]
    ] = dataclasses.field(default=None, init=False)
    _model_composer: Optional[BaseModelComposer] = dataclasses.field(default=None, init=False)
    _model_selection: Optional[BestModelSelection] = dataclasses.field(default=None, init=False)
    _normalization_statistics: Optional[StructuredStatisticsType] = dataclasses.field(default=None, init=False)
    _trainer: Optional[Trainer] = dataclasses.field(default=None, init=False)
    _cfg: Optional[DictConfig] = dataclasses.field(default=None, init=False)

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up prerequisites to training.
        Includes wrapping the environment for obseration normalization, instantiating the model composer etc.
        :param cfg: DictConfig defining components to initialize.
        """

        self._cfg = cfg

        # Generate a random state used for sampling random seeds for the envs and agents
        self.maze_seeding = MazeSeeding(cfg.seeding.env_base_seed, cfg.seeding.agent_base_seed,
                                        cfg.seeding.cudnn_determinism_flag)

        with SwitchWorkingDirectoryToInput(cfg.input_dir):
            assert isinstance(cfg.env, DictConfig) or isinstance(cfg.env, Callable)
            wrapper_cfg = cfg.wrappers if "wrappers" in cfg else {}

            # if the observation normalization is already available, read it from the input directory
            if isinstance(cfg.env, DictConfig):
                self.env_factory = EnvFactory(cfg.env, wrapper_cfg)
            elif isinstance(cfg.env, Callable):
                self.env_factory = lambda: WrapperFactory.wrap_from_config(cfg.env(), wrapper_cfg)

            normalization_env = self.env_factory()
            normalization_env.seed(self.maze_seeding.generate_env_instance_seed())

        # Observation normalization
        self._normalization_statistics = obtain_normalization_statistics(normalization_env,
                                                                         n_samples=self.normalization_samples)
        if self._normalization_statistics:
            self.env_factory = make_normalized_env_factory(self.env_factory, self._normalization_statistics)
            # dump statistics to current working directory
            assert isinstance(normalization_env, ObservationNormalizationWrapper)
            normalization_env.dump_statistics()

        # Generate an agent seed and set the seed globally for the model initialization
        set_seeds_globally(self.maze_seeding.agent_global_seed, self.maze_seeding.cudnn_determinism_flag,
                           info_txt=f'training runner (Pid:{os.getpid()})')

        # init model composer
        composer_type = Factory(base_type=BaseModelComposer).type_from_name(cfg.model['_target_'])
        composer_type.check_model_config(cfg.model)

        # todo Factory.instantiate returns specified dicts as DictConfig, i.e. many specified types are wrong. How do we
        #  go about this? DictConfig behaves similarly to Dict for all intents and purposes, but typing is still off/
        #  misleading. This is independent from our Python training API and can apparently not be changed, i.e. kwargs
        #  seems to be always converted to DictConfig/ListConfig.
        self._model_composer = Factory(base_type=BaseModelComposer).instantiate(
            cfg.model,
            action_spaces_dict=normalization_env.action_spaces_dict,
            observation_spaces_dict=normalization_env.observation_spaces_dict,
            agent_counts_dict=normalization_env.agent_counts_dict)

        SpacesConfig(self._model_composer.action_spaces_dict,
                     self._model_composer.observation_spaces_dict,
                     self._model_composer.agent_counts_dict).save(self.spaces_config_dump_file)

        # Should be done after the normalization runs, otherwise stats from those will get logged as well.
        setup_logging(job_config=cfg)

        # close normalization env
        normalization_env.close()

    def run(self, n_epochs: Optional[int] = None, **train_kwargs) -> None:
        """
        Runs training.
        While this method is designed to be overriden by individual subclasses, it provides some functionality
        that is useful in general:

        - Building the env factory for env + wrappers
        - Estimating normalization statistics from the env
        - If successfully estimated, wrapping the env factory so that envs are already built with the statistics
        - Building the model composer from model config and env spaces config
        - Serializing the env spaces configuration (so that the model composer can be re-loaded for future rollout)
        - Initializing logging setup

        :param n_epochs: Number of epochs to train.
        :param train_kwargs: Additional arguments for trainer.train().
        """

        self._trainer.train(
            n_epochs=self._cfg.algorithm.n_epochs if n_epochs is None else n_epochs,
            **train_kwargs
        )

    @classmethod
    def _init_trainer_from_input_dir(cls, trainer: Trainer, state_dict_dump_file: str, input_dir: str) -> None:
        """Initialize trainer from given state dict and input directory.
        :param trainer: The trainer to initialize.
        :param state_dict_dump_file: The state dict dump file relative to input_dir.
        :param input_dir: The directory to load the state dict from.
        """
        with SwitchWorkingDirectoryToInput(input_dir):
            if os.path.exists(state_dict_dump_file):
                BColors.print_colored(
                    f"Trainer and model initialized from '{state_dict_dump_file}' of run '{input_dir}'!",
                    BColors.OKGREEN)
                trainer.load_state(state_dict_dump_file)
            else:
                BColors.print_colored("Model initialized with random weights! ", BColors.OKGREEN)

    @property
    def model_composer(self) -> BaseModelComposer:
        """
        Returns model composer.
        :return: Model composer.
        """

        return self._model_composer
