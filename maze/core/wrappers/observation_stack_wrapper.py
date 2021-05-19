""" Implements observation stacking as an environment wrapper. """
import copy
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
from gym import spaces
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.utils.structured_env_utils import flat_structured_space
from maze.core.wrappers.wrapper import ObservationWrapper, Wrapper


class ObservationStackWrapper(ObservationWrapper[MazeEnv]):
    """An wrapper stacking the observations of multiple subsequent time steps.

    Provides functionality for:

        - selecting which observations to stack
        - how many past observations should be stacked
        - stacking deltas with the current step observation (instead of the observations itself)

    :param env: Environment/wrapper to wrap.
    :param stack_config: The observation stacking configuration.

        observation:        The name (key) of the respective observation
        keep_original:      Bool, indicates weather to keep or remove the original observation from the dictionary.
        tag:                Optional[str], tag to add to observation (e.g. stacked)
        delta:              Bool, if true deltas are stacked to the previous observation
        stack_steps:        Int, number of past steps to be stacked
    """

    def __init__(self, env: StructuredEnvSpacesMixin, stack_config: List[Dict[str, Any]]):
        super().__init__(env)

        self.stack_config = stack_config
        self.drop_original = False

        # initialize observation collection and statistics
        self._original_observation_spaces_dict = copy.deepcopy(env.observation_spaces_dict)

        # flatten the structured observation space
        self._flat_observation_space = flat_structured_space(self._original_observation_spaces_dict)

        # Initialize normalization strategies for all sub step hierarchies and observations
        self._initialize_stacking()

        # initialize observation stack
        self.max_steps = max([c["stack_steps"] for c in self.stack_config])
        self._observation_stack: Dict[str, List[np.ndarray]] = defaultdict(list)

    @override(ObservationWrapper)
    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Stack observations.

        :param observation: The observation to be stacked.
        :return: The sacked observation.
        """

        # iteratively process observations
        for config in self.stack_config:

            # extract config
            obs_key = config["observation"]
            keep_original = config["keep_original"]
            tag = config["tag"]
            delta = config["delta"]
            stack_steps = config["stack_steps"]

            # update observation stack
            if obs_key in observation:
                self._observation_stack[obs_key].append(observation[obs_key])
                self._observation_stack[obs_key] = self._observation_stack[obs_key][-self.max_steps:]

            # nothing to do
            if stack_steps < 2 or obs_key not in observation:
                continue

            # compute delta stack
            if delta:
                stacked = np.stack(self._observation_stack[obs_key])
                if stacked.shape[0] > 1:
                    diff = np.diff(stacked, axis=0)
                    stacked[-(len(diff) + 1):-1] = diff

            # just stack
            else:
                stacked = np.stack(self._observation_stack[obs_key])

            # pad with zeros if necessary
            shape = [stack_steps] + list(self._flat_observation_space[obs_key].shape)
            observation_stack = np.zeros(shape=shape, dtype=np.float32)
            observation_stack[-len(stacked):] = stacked

            # drop original observation
            if not keep_original:
                del observation[obs_key]

            # put observation into dictionary
            full_tag = obs_key if tag is None else f"{obs_key}-{tag}"
            observation[full_tag] = observation_stack

        return observation

    @override(ObservationWrapper)
    def reset(self) -> Dict[str, np.ndarray]:
        """Intercept ``ObservationWrapper.reset`` and map observation."""
        # reset observation stack
        self._observation_stack: Dict[str, List[np.ndarray]] = defaultdict(list)
        return super().reset()

    def _initialize_stacking(self) -> None:
        """Initialize observation stacking for all sub steps and all dictionary observations.
        """

        # iterate stacking config
        for mapping in self.stack_config:
            obs_key = mapping["observation"]
            assert obs_key in self._flat_observation_space.spaces, \
                f"Observation {obs_key} not contained in flat observation space."

            # iterate all structured env sub steps and update observation spaces accordingly
            for sub_step_key, sub_space in self._original_observation_spaces_dict.items():
                if obs_key in sub_space.spaces:

                    # nothing to stack
                    stack_steps = mapping["stack_steps"]
                    if stack_steps < 2:
                        continue

                    # get current space
                    cur_space = self.observation_spaces_dict[sub_step_key][obs_key]

                    # compute stacked low / high
                    if mapping["delta"]:
                        float_max = np.finfo(np.float32).max
                        float_min = np.finfo(np.float32).min

                        # compute delta lows and highs
                        mask = cur_space.low > float_min
                        delta_min = np.full(cur_space.low.shape, fill_value=float_min, dtype=cur_space.dtype)
                        delta_min[mask] = np.clip(cur_space.low[mask] - cur_space.high[mask], float_min, None)

                        mask = cur_space.high < float_max
                        delta_max = np.full(cur_space.high.shape, fill_value=float_max, dtype=cur_space.dtype)
                        delta_max[mask] = np.clip(cur_space.high[mask] + cur_space.high[mask], None, float_max)

                        low = np.stack([delta_min] * (stack_steps - 1) + [cur_space.low])
                        high = np.stack([delta_max] * (stack_steps - 1) + [cur_space.high])
                    else:
                        low = np.stack([cur_space.low] * stack_steps)
                        high = np.stack([cur_space.high] * stack_steps)

                    # remove original key from observation space
                    if not mapping["keep_original"]:
                        self.observation_spaces_dict[sub_step_key].spaces.pop(obs_key)

                    # add stacked observation space
                    full_tag = obs_key if mapping["tag"] is None else f"{obs_key}-{mapping['tag']}"
                    new_space = spaces.Box(low=low, high=high, shape=None, dtype=cur_space.dtype)
                    self.observation_spaces_dict[sub_step_key].spaces[full_tag] = new_space
                    assert cur_space.low.ndim == (new_space.low.ndim - 1)

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """If this is the first step in an episode, reset the observation stack."""
        if first_step_in_episode:
            self._observation_stack: Dict[str, List[np.ndarray]] = defaultdict(list)

        return super().get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'ObservationStackWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self._observation_stack = copy.deepcopy(env._observation_stack)
        self.env.clone_from(env)
