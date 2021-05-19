""" Implements observation normalization as an environment wrapper."""
import copy
import os
import pickle
from collections import defaultdict
from types import ModuleType
from typing import Any, Dict, Union, List, Optional, Tuple

import gym
import numpy as np
from maze.core.env.action_conversion import ActionType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from omegaconf import DictConfig

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.env.structured_env import StructuredEnv
from maze.core.utils.factory import Factory
from maze.core.wrappers.observation_normalization.normalization_strategies.base import \
    ObservationNormalizationStrategy, StructuredStatisticsType
from maze.core.wrappers.wrapper import ObservationWrapper


class ObservationNormalizationWrapper(ObservationWrapper[MazeEnv]):
    """An observation normalization wrapper.
    It provides functionality for:

        - normalizing observations according to specified normalization strategies
        - clipping observations according to specified min and max values
        - estimating normalization statistics from observations collecting by interacting with the environment
        - manually overwriting the observation normalization parameters

    The current implementation assumes that observation space is **always** a Dict (even if just a Dict-wrapped Box).

    :param env: Environment/wrapper to wrap.
    :param default_strategy: The default observation normalization strategy.
    :param default_statistics: Manual default normalization statistics.
    :param statistics_dump: Path to a pickle file dump of normalization statistics.
    :param sampling_policy: The sampling policy for estimating the statistics.
    :param exclude: List of observation keys to exclude from normalization.
    :param manual_config: Additional manual configuration options.
    """

    @classmethod
    def register_new_observation_normalization_strategy(cls, containing_submodule: Any):
        """Registers a new observation normalization strategy.

        :param containing_submodule: Add all classes implementing ObservationNormalizationStrategy by walking the module
                                     recursively.
        """

        # Note: Modules are of type ModuleType. PyCharm doesn't recognize this for type hints however, so we rely on Any
        # + type assertion.
        assert isinstance(containing_submodule, ModuleType)

        cls.registry_normalization_strategies.collect_modules(
            root_module=containing_submodule,
            base_type=ObservationNormalizationStrategy
        )

    def __init__(self, env: MazeEnv,
                 default_strategy: Union[str, ObservationNormalizationStrategy],
                 default_strategy_config: Dict[str, Any],
                 default_statistics: Optional[Dict[str, Any]],
                 statistics_dump: str,
                 sampling_policy: Union[DictConfig, Policy],
                 exclude: Optional[List[str]],
                 manual_config: Optional[Dict[Union[str, int], Dict[str, Any]]]):
        super().__init__(env)

        self.default_strategy = default_strategy
        self.default_strategy_config = default_strategy_config
        self.default_statistics = default_statistics
        self.statistics_dump = statistics_dump
        self.exclude = [] if exclude is None else exclude
        self.manual_config = manual_config
        self.sampling_policy: Policy = \
            Factory(Policy).instantiate(sampling_policy, action_spaces_dict=env.action_spaces_dict)

        # initialize observation collection and statistics
        self._original_observation_spaces_dict = copy.deepcopy(env.observation_spaces_dict)
        self._collect_observations: bool = False
        self._collected_observation: Dict[str, List[np.ndarray]] = defaultdict(list)

        # load statistics dump
        self.loaded_stats: StructuredStatisticsType = defaultdict(dict)
        if self.statistics_dump is not None and os.path.exists(self.statistics_dump):
            self.loaded_stats = self._load_statistics()

        # Initialize normalization strategies for all sub step hierarchies and observations
        self._normalization_strategies: Dict[str, ObservationNormalizationStrategy] = defaultdict()
        self._initialize_normalization_strategies()

    @override(StructuredEnv)
    def seed(self, seed: int) -> None:
        """Apply seed to wrappers rng, and pass the seed forward to the env
        """
        self.sampling_policy.seed(seed)
        return self.env.seed(seed)

    @override(ObservationWrapper)
    def observation(self, observation: Any) -> Any:
        """Collect observations for statistics computation or normalize them.

        :param observation: The observation to be normalized.
        :return: The normalized observation.
        """

        # collect observations for statistics computation
        if self._collect_observations:
            self._collect_observation(observation)

        # normalize dictionary observation
        else:
            observation = copy.deepcopy(observation)
            for obs_key in observation:
                if obs_key not in self.exclude:
                    strategy = self._normalization_strategies[obs_key]
                    assert strategy.is_initialized(), \
                        f"Normalization statistics are not properly initialized for '{obs_key}'!"
                    observation[obs_key] = strategy.normalize_and_process_value(observation[obs_key])

        return observation

    def set_observation_collection(self, status: bool) -> None:
        """Activate / deactivate observation collection.

        :param status: If True observations are collected for statistics computation.
                       If False observations are normalized with the provided statistics
        """
        self._collect_observations = status

        # reset collected observations
        if status is False:
            self._collected_observation = defaultdict(list)

    def get_statistics(self) -> StructuredStatisticsType:
        """Returns the normalization statistics of the respective normalization strategy.
        :return: The normalization statistics for all sub steps and all dictionary observations.
        """
        statistics = defaultdict()
        for sub_step_key in self.observation_spaces_dict:
            sub_space_dict = self.observation_spaces_dict[sub_step_key].spaces
            for observation in sub_space_dict:
                if observation not in self.exclude:
                    norm_strategy = self._normalization_strategies[observation]
                    if observation not in statistics:
                        statistics[observation] = norm_strategy.get_statistics()
        return statistics

    def estimate_statistics(self) -> None:
        """Estimates and sets the observation statistics from collected observations.
        """

        # list of observations with statistics
        has_statistics = [key for key, stats in self.get_statistics().items() if stats is not None]

        # iterate structured env sub steps
        for sub_step_key in self._original_observation_spaces_dict:
            sub_space = self._original_observation_spaces_dict[sub_step_key]
            assert isinstance(sub_space, gym.spaces.Dict)

            # iterate observation of sub steps
            for obs_key in sub_space.spaces:

                # no need to estimate stats
                if obs_key in self.exclude:
                    continue
                # stats already set
                if obs_key in has_statistics:
                    continue

                strategy = self._normalization_strategies[obs_key]
                collected_obs = self._collected_observation[obs_key]

                # estimate and set statistics if no manual stats are specified
                if not self._has_manual_config_key(obs_key, "statistics"):
                    stats = strategy.estimate_stats(collected_obs)
                    strategy.set_statistics(stats)

                    # update the observation space accordingly
                    self.observation_spaces_dict[sub_step_key].spaces[obs_key] = strategy.normalized_space()

        # dump statistics to file
        self.dump_statistics()

    def set_normalization_statistics(self, stats: StructuredStatisticsType) -> None:
        """Apply existing normalization statistics.

        :param stats: The statistics dict
        """
        self.loaded_stats = stats

        # Initialize normalization strategies for all sub step hierarchies and observations
        self._normalization_strategies: Dict[str, ObservationNormalizationStrategy] = defaultdict()
        self._initialize_normalization_strategies()

    def dump_statistics(self) -> None:
        """Dump statistics to file.
        """
        all_stats = self.get_statistics()
        with open(self.statistics_dump, "wb") as fp:
            pickle.dump(all_stats, fp)

    def _load_statistics(self) -> StructuredStatisticsType:
        """Load statistics from file.

        :return: Statistics loaded from dump file.
        """
        with open(self.statistics_dump, "rb") as fp:
            dumped_statistics = pickle.load(fp)
        return dumped_statistics

    def _initialize_normalization_strategies(self) -> None:
        """Initialize normalization strategies for all sub steps and all dictionary observations.
        """
        # iterate sub steps
        for sub_step_key, sub_space in self._original_observation_spaces_dict.items():
            assert isinstance(sub_space, gym.spaces.Dict), "Only gym.spaces.Dict are supported as of now!"

            # iterate keys of dict observation space
            for obs_key in sub_space.spaces.keys():

                if obs_key in self.exclude:
                    continue

                # start out with default values
                normalization_strategy = self.default_strategy
                strategy_config = copy.copy(self.default_strategy_config)
                statistics = self.default_statistics

                # check if statistics have been computed and dumped
                if obs_key in self.loaded_stats:
                    statistics = self.loaded_stats[obs_key]

                # check if a manual config is specified
                if self._has_manual_config(obs_key):
                    manual_obs_config = self.manual_config[obs_key]

                    normalization_strategy = manual_obs_config.get("strategy", normalization_strategy)
                    statistics = manual_obs_config.get("statistics", statistics)
                    strategy_config.update(manual_obs_config.get("strategy_config", dict()))

                # build normalization strategy
                strategy = Factory(ObservationNormalizationStrategy).instantiate({
                    "_target_": normalization_strategy,
                    "observation_space": sub_space[obs_key],
                    **strategy_config
                })

                # update the observation space accordingly
                if statistics is not None and obs_key not in self.exclude:
                    strategy.set_statistics(statistics)
                    self.observation_spaces_dict[sub_step_key].spaces[obs_key] = strategy.normalized_space()

                self._normalization_strategies[obs_key] = strategy

        # make sure that everything has been applied properly
        if self.manual_config is not None:
            self._check_manual_config()

    def _check_manual_config(self) -> None:
        """Check if manual configuration has been applied properly.
        """
        # iterate keys of dict observation space
        for obs_key in self.manual_config:
            assert obs_key in self._normalization_strategies, \
                f"Normalization of observation '{obs_key}' was not initialized properly!"

    def _collect_observation(self, observation: Dict[str, np.ndarray]) -> None:
        """Collect observations for normalization statistics computation.

        :param observation: The observation to collect.
        """
        for key in observation.keys():
            self._collected_observation[key].append(observation[key][np.newaxis])

    def _has_manual_config(self, observation_key) -> bool:
        """Checks if the observation space path has a manual config provided.

        :param observation_key: The dictionary space observation key.
        :return: True if manual config provided; else False
        """
        has_manual_config = self.manual_config is not None and observation_key in self.manual_config
        return has_manual_config

    def _has_manual_config_key(self, observation_key, config_key) -> bool:
        """Checks if the observation space path has a manual config and a selected key provided.

        :param observation_key: The dictionary space observation key.
        :param config_key: The config key to check for.
        :return: True if the manual config contains the selected key provided; else False
        """
        has_manual_config_key = self._has_manual_config(observation_key) and \
                                config_key in self.manual_config[observation_key]

        return has_manual_config_key

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'ObservationNormalizationWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
