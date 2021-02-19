"""Implements the default random policy for structured envs by sampling from the action_space."""
from typing import Union, Dict, Tuple, Sequence, Optional

from gym import spaces
from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.utils.config_utils import make_env
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper
from omegaconf import DictConfig


class RandomPolicy(Policy):
    """Implements a random structured policy.

    :param action_spaces_dict: The action_spaces dict from the env
    """

    def __init__(self, action_spaces_dict: Dict[Union[str, int], spaces.Space]):
        self.action_spaces_dict = action_spaces_dict

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state() object to compute the action."""
        return False

    @override(Policy)
    def compute_action(self, observation: ObservationType, maze_state: Optional[MazeStateType],
                       policy_id: Union[str, int] = None, deterministic: bool = False) -> ActionType:
        """
        Query a policy that corresponds to the given ID for action.

        :param observation: Current observation of the environment
        :param maze_state: Current state of the environment (will always be None as `needs_state()` returns False)
        :param policy_id: ID of the policy to query (does not have to be provided if policies dict contain only 1 policy
        :param deterministic: Specify if the action should be computed deterministically
        :return: Next action to take
        """
        return self.action_spaces_dict[policy_id].sample()

    @override(Policy)
    def compute_top_action_candidates(self, observation: ObservationType,
                                      num_candidates: int, maze_state: Optional[MazeStateType] = None,
                                      policy_id: Union[str, int] = None, deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface
        """


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
