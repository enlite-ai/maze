"""This file holds the rllib runner, which uses the maze model-builder as well as the maze distribution
    mapper in order to train the maze env with RLlib algorithms"""
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import ray
from omegaconf import DictConfig, OmegaConf
from ray import tune
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS

from maze.core.annotations import override
from maze.core.utils.factory import Factory
from maze.core.utils.seeding import MazeSeeding
from maze.rllib.maze_rllib_action_distribution import MazeRLlibActionDistribution
from maze.rllib.maze_rllib_callback_logging import MazeRLlibLoggingCallbacks
from maze.rllib.maze_rllib_env import build_maze_rllib_env_factory
from maze.rllib.maze_rllib_models.maze_rllib_base_model import MazeRLlibBaseModel
from maze.rllib.maze_tune_callback_save_model import MazeRLlibSaveModelCallback
from maze.runner import Runner


@dataclass
class MazeRLlibRunner(Runner):
    """Base class for rllib runner

    :param spaces_config_dump_file: The path to the spaces config dump file.
    :param normalization_samples: The number of normalization samples that should be computed in order to estimated
        observation statistics.
    :param num_workers: The number of worker that should be used.
    :param tune_config: The tune config, used as arguments when starting tune.run().
    :param ray_config: The ray config, used as arguments when initializing ray (ray.init()).
    :param state_dict_dump_file: The path to the state dict dump file.
    """

    normalization_samples: int
    """Number of samples (=steps) to collect normalization statistics at the beginning of the training."""

    spaces_config_dump_file: str
    """Where to save the env spaces configuration (output directory handled by hydra)"""

    num_workers: int
    """Specifies the number of workers for the ray rllib distribution"""

    tune_params: Dict[str, Any]
    """Specify the parameters for the ray.tune method"""

    ray_config: Dict[str, Any]
    """Specify the parameters for the ray.init method"""

    state_dict_dump_file: str
    """Where to save the best model (output directory handled by hydra)"""

    def __init__(self, spaces_config_dump_file: str, normalization_samples: int,
                 num_workers: int, tune_config: Dict[str, Any], ray_config: Dict[str, Any],
                 state_dict_dump_file: str):
        self.spaces_config_dump_file = spaces_config_dump_file
        self.normalization_samples = normalization_samples
        self.num_workers = num_workers
        self.ray_config = ray_config
        self.tune_params = tune_config
        self.state_dict_dump_file = state_dict_dump_file

        self.env_factory = None
        self.model_composer = None
        self.normalization_statistics = None

        self._cfg: Optional[DictConfig] = None
        self.rllib_config: Optional[Dict[str, Any]] = None
        self.tune_config: Optional[Dict[str, Any]] = None

    def setup(self, cfg: DictConfig):
        """
        This method initializes and registers all necessary maze components with RLlib

        :param cfg: Full Hydra run job config
        """
        # Generate a random state used for sampling random seeds for the envs and agents
        self.maze_seeding = MazeSeeding(cfg.seeding.env_base_seed, cfg.seeding.agent_base_seed,
                                        cfg.seeding.cudnn_determinism_flag)

        self._cfg = cfg

        # Initialize env factory (with rllib monkey patches)
        self.env_factory = build_maze_rllib_env_factory(cfg)

        # Register maze env factory with rllib
        tune.register_env("maze_env", lambda x: self.env_factory())

        # Register maze model and distribution mapper if a maze model should be used
        # Check whether we are using the rllib default model composer or a maze model
        using_rllib_model_composer = '_target_' not in cfg.model.keys()
        if not using_rllib_model_composer:
            # Get model class
            model_cls = Factory(MazeRLlibBaseModel).type_from_name(cfg.algorithm.model_cls)
            # Register maze model
            ModelCatalog.register_custom_model("maze_model", model_cls)

            if 'policy' in cfg.model and "networks" in cfg.model.policy:
                assert len(cfg.model.policy.networks) == 1, 'Hierarchical envs are not yet supported'

            # register maze action distribution
            ModelCatalog.register_custom_action_dist('maze_dist', MazeRLlibActionDistribution)
            model_config = {
                "custom_action_dist": 'maze_dist',
                "custom_model": "maze_model",
                "vf_share_layers": False,
                "custom_model_config": {
                    "maze_model_composer_config": cfg.model,
                    'spaces_config_dump_file': self.spaces_config_dump_file,
                    'state_dict_dump_file': self.state_dict_dump_file
                }
            }
        else:
            # If specified use the default rllib model builder
            model_config = OmegaConf.to_container(cfg.model, resolve=True)

        # Build rllib config
        maze_rllib_config = {
            "env": "maze_env",
            # Store env config for possible later use
            "env_config": {'env': cfg.env, 'wrappers': cfg.wrappers},
            "model": model_config,
            'callbacks': MazeRLlibLoggingCallbacks,
            "framework": "torch"
        }
        # Load the algorithm config and update the custom parameters
        rllib_config: Dict = OmegaConf.to_container(cfg.algorithm.config, resolve=True)
        assert 'model' not in rllib_config, 'The config should be removed from the default yaml files since it will ' \
                                            'be dynamically written'
        assert self.num_workers == rllib_config['num_workers']
        rllib_config.update(maze_rllib_config)

        if rllib_config['seed'] is None:
            rllib_config['seed'] = self.maze_seeding.generate_env_instance_seed()

        # Initialize ray with the passed ray_config parameters
        ray_config: Dict = OmegaConf.to_container(self.ray_config, resolve=True)

        # Load tune parameters
        tune_config = OmegaConf.to_container(self.tune_params, resolve=True)
        tune_config['callbacks'] = [MazeRLlibSaveModelCallback()]

        # Start tune experiment
        assert 'config' not in tune_config, 'The config should be removed from the default yaml files since it will ' \
                                            'be dynamically written'

        self.ray_config = ray_config
        self.rllib_config = rllib_config
        self.tune_config = tune_config

    @override(Runner)
    def run(self) -> None:
        """
        This method initializes and registers all necessary maze components with RLlib before initializing ray and
        starting a tune.run with the config parameters.
        """

        # Init ray
        ray.init(**self.ray_config)

        # Run tune
        tune.run(self._cfg.algorithm.algorithm, config=self.rllib_config, **self.tune_config)

        # Shutdown ray
        ray.shutdown()
