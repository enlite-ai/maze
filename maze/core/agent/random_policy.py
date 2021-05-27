"""Implements the default random policy for structured envs by sampling from the action_space."""
from typing import Union, Dict, Tuple, Sequence, Optional

import numpy as np
from gym import spaces
from omegaconf import DictConfig

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.utils.config_utils import make_env
from maze.core.utils.seeding import MazeSeeding
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper
from maze.train.utils.train_utils import stack_numpy_dict_list


class RandomPolicy(Policy):
    """Implements a random structured policy.

    :param action_spaces_dict: The action_spaces dict of the env (will sample from it).
    """

    def __init__(self, action_spaces_dict: Dict[Union[str, int], spaces.Space]):
        self.action_spaces_dict = dict(action_spaces_dict)

    @override(Policy)
    def seed(self, seed: int) -> None:
        """Seed the policy by setting the action space seeds."""
        rng = np.random.RandomState(seed)
        for key, action_space in self.action_spaces_dict.items():
            action_space.seed(MazeSeeding.generate_seed_from_random_state(rng))
        pass

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state() object to compute the action."""
        return False

    @override(Policy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[MazeStateType],
                       env: Optional[BaseEnv] = None,
                       actor_id: Optional[ActorID] = None,
                       deterministic: bool = False) -> ActionType:
        """Sample random action from the given action space."""
        if actor_id:
            action_space = self.action_spaces_dict[actor_id.step_key]
        else:
            assert len(self.action_spaces_dict) == 1, "action spaces for multiple sub-steps are available, please " \
                                                      "specify actor ID explicitly"
            action_space = list(self.action_spaces_dict.values())[0]

        return action_space.sample()

    @override(Policy)
    def compute_top_action_candidates(self,
                                      observation: ObservationType,
                                      num_candidates: Optional[int],
                                      maze_state: Optional[MazeStateType] = None,
                                      env: Optional[BaseEnv] = None,
                                      actor_id: ActorID = None,
                                      deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """Random policy does not provide top action candidates."""


class DistributedRandomPolicy(RandomPolicy):
    """Implements random structured policy with the ability to sample multiple actions simultaneously.

    Mimics behavior of a Torch policy in batched distributed environment scenarios,
    when actions are sampled for all environments at once. (Useful mostly for testing purposes.)

    :param action_spaces_dict: The action_spaces dict of the env (will sample from it)
    :param concurrency: How many actions to sample at once. Should correspond to concurrency of the distributed env.
    """

    def __init__(self, action_spaces_dict: Dict[Union[str, int], spaces.Space], concurrency: int):
        super().__init__(action_spaces_dict)
        self.concurrency = concurrency

    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[MazeStateType],
                       env: Optional[BaseEnv] = None,
                       actor_id: Optional[ActorID] = None,
                       deterministic: bool = False) -> ActionType:
        """Sample multiple actions together."""
        if actor_id:
            action_space = self.action_spaces_dict[actor_id.step_key]
        else:
            assert len(self.action_spaces_dict) == 1, "action spaces for multiple sub-steps are available, please " \
                                                      "specify actor ID explicitly"
            action_space = list(self.action_spaces_dict.values())[0]

        return stack_numpy_dict_list([action_space.sample() for _ in range(self.concurrency)])


def random_policy_from_config(env_config: DictConfig, wrappers_config: DictConfig) -> RandomPolicy:
    """Factory function for instantiating a :class:`~maze.core.agent.structured_policy.RandomPolicy`
    from an env configuration.

    :param env_config: Environment config
    :param wrappers_config: Wrapper config
    :return: A random structured policy.
    """
    env = make_env(env_config, wrappers_config)
    # observation normalization needs to be deactivated when no stats are provided
    if isinstance(env, ObservationNormalizationWrapper):
        env.set_observation_collection(True)
    return RandomPolicy(env.action_spaces_dict)
