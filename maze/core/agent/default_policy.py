"""Default implementation of structured policy."""

from typing import Union, Tuple, Sequence, Optional

from maze.core.agent.flat_policy import FlatPolicy
from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.utils.registry import Registry, CollectionOfConfigType


class DefaultPolicy(Policy):
    """Encapsulates one or more policies identified by policy IDs.

    :param policies: Dict of policy IDs and corresponding policies.
    """

    def __init__(self, policies: CollectionOfConfigType):
        self.policies = Registry(FlatPolicy).arg_to_collection(policies)

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state() object to compute the action."""
        return False

    @override(Policy)
    def compute_action(self, observation: ObservationType, maze_state: Optional[MazeStateType] = None,
                       policy_id: Union[str, int] = None, deterministic: bool = False) -> ActionType:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface
        """
        if policy_id is None:
            assert len(self.policies.items()) == 1, "no policy ID provided but multiple policies are available"
            return list(self.policies.values())[0].compute_action(observation, deterministic=deterministic)
        else:
            return self.policies[policy_id].compute_action(observation, deterministic=deterministic)

    @override(Policy)
    def compute_top_action_candidates(self, observation: ObservationType,
                                      num_candidates: int, maze_state: Optional[MazeStateType] = None,
                                      policy_id: Union[str, int] = None, deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface
        """
        raise NotImplementedError
