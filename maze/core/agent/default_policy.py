"""Default implementation of structured policy."""

from typing import Tuple, Sequence, Optional

from maze.core.agent.flat_policy import FlatPolicy
from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.utils.factory import Factory, CollectionOfConfigType


class DefaultPolicy(Policy):
    """Encapsulates one or more policies identified by policy IDs.

    :param policies: Dict of policy IDs and corresponding policies.
    """

    def __init__(self, policies: CollectionOfConfigType):
        self.policies = Factory(FlatPolicy).instantiate_collection(policies)

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state() object to compute the action."""
        return False

    @override(Policy)
    def seed(self, seed: int) -> None:
        """Not applicable since Global seed should already be set before initializing the models"""
        pass

    @override(Policy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[MazeStateType] = None,
                       env: Optional[BaseEnv] = None,
                       actor_id: Optional[ActorID] = None,
                       deterministic: bool = False) -> ActionType:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface"""
        return self.policy_for(actor_id).compute_action(observation, deterministic=deterministic)

    def policy_for(self, actor_id: Optional[ActorID]) -> FlatPolicy:
        """Return policy corresponding to the given actor ID (or the single available policy if no actor ID is provided)

        :param actor_id: Actor ID to get policy for
        :return: Flat policy corresponding to the actor ID
        """
        if actor_id is None:
            assert len(self.policies.items()) == 1, "no policy ID provided but multiple policies are available"
            return list(self.policies.values())[0]
        else:
            return self.policies[actor_id.step_key]

    @override(Policy)
    def compute_top_action_candidates(self,
                                      observation: ObservationType,
                                      num_candidates: Optional[int],
                                      maze_state: Optional[MazeStateType] = None,
                                      env: Optional[BaseEnv] = None,
                                      actor_id: Optional[ActorID] = None,
                                      deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface"""
        raise NotImplementedError
