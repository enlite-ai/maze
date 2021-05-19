""" Implements observation pre-processing as an observation wrapper. """
from typing import Any, Dict, List, Tuple, Mapping

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.utils.factory import Factory
from maze.core.utils.structured_env_utils import flat_structured_space
from maze.core.wrappers.observation_preprocessing.preprocessors.base import PreProcessor
from maze.core.wrappers.wrapper import ObservationWrapper


class PreProcessingWrapper(ObservationWrapper[MazeEnv]):
    """An observation pre-processing wrapper.
    It provides functionality for:

    - pre-processing observations (flattening, one-hot encoding, ...)
    - adopting the observation spaces accordingly

    :param env: Environment/wrapper to wrap.
    :param pre_processor_mapping: The pre-processing configuration.
           Example mappings can be found in our reference documentation.
    """

    def __init__(self, env: StructuredEnvSpacesMixin, pre_processor_mapping: List[Dict[str, Any]]):
        super().__init__(env)

        self.pre_processor_mapping = pre_processor_mapping
        self.drop_original = False

        # Initialize normalization strategies for all sub step hierarchies and observations
        self._preprocessors: List[Tuple[str, PreProcessor, bool]] = list()
        self._initialize_preprocessors()

    @override(ObservationWrapper)
    def observation(self, observation: Any) -> Any:
        """Pre-processes observations.

        :param observation: The observation to be pre-processed.
        :return: The pre-processed observation.
        """

        # iteratively pre-process observations
        for obs_key, processor, keep_original in self._preprocessors:

            # check if obs_key is in the observation
            if obs_key in observation:
                tag = f"{obs_key}-{processor.tag()}"
                observation[tag] = processor.process(observation=observation[obs_key])

                # drop original observation
                if not keep_original:
                    del observation[obs_key]

        return observation

    def _initialize_preprocessors(self) -> None:
        """Initialize pre-processors for all sub steps and all dictionary observations.
        """

        # get full flat observation space
        observation_spaces = flat_structured_space(self.observation_spaces_dict).spaces

        # maintain a list of temporary spaces
        temporary_spaces = []

        # iterate pre-processor config
        for mapping in self.pre_processor_mapping:
            obs_key = mapping["observation"]
            assert obs_key in observation_spaces, f"Observation {obs_key} not contained in observation space."

            pre_processor_cls = Factory(PreProcessor).type_from_name(mapping["_target_"])
            assert isinstance(mapping["config"], Mapping), \
                f"Make sure that the config for {pre_processor_cls.__name__} of observation {obs_key} is a dict!"
            processor = pre_processor_cls(observation_space=observation_spaces[obs_key], **mapping["config"])

            self._preprocessors.append((obs_key, processor, mapping["keep_original"]))

            # append processed space
            tag = f"{obs_key}-{processor.tag()}"
            observation_spaces[tag] = processor.processed_space()

            # iterate all structured env sub steps and update observation spaces accordingly
            for sub_step_key, sub_space in self.observation_spaces_dict.items():

                # check if the subspace is contained
                if obs_key in sub_space.spaces:

                    # add new key to observation space
                    self.observation_spaces_dict[sub_step_key].spaces[tag] = processor.processed_space()

                    # remove original key from observation space
                    if not mapping["keep_original"]:
                        temporary_spaces.append((sub_step_key, obs_key))

        # remove temporary spaces
        for sub_step_key, obs_key in temporary_spaces:
            self.observation_spaces_dict[sub_step_key].spaces.pop(obs_key)

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'PreProcessingWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
