from typing import Dict, Sequence, Tuple, Union, Optional

import numpy as np

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.utils.factory import Factory, ConfigType
from maze_envs.logistics.cutting_2d.env.maze_action import Cutting2DMazeAction
from maze_envs.logistics.cutting_2d.space_interfaces.action_conversion.base import BaseActionConversion


class GreedyPolicy(Policy):
    """
    Heuristic agent. Always picks the smallest piece possible for cutting (by area).

    :param action_conversion: ActionConversionInterface.
    """

    def __init__(self, action_conversion: ConfigType):
        super().__init__()
        self.action_conversion = Factory(base_type=BaseActionConversion).instantiate(action_conversion)

    @override(Policy)
    def seed(self, seed: int) -> None:
        """Not applicable since heuristic is deterministic"""
        pass

    def get_candidate_pieces(self, observation: Dict) -> Sequence[Cutting2DMazeAction]:
        """
        Go through all of the pieces in inventory and select all possible candidates,
        ranked from the best (= smallest piece).

        Currently goes through the whole inventory -- could be further optimized if needed.

        :param observation: Observation of the environment
        :return: List of candidate MazeActions ranked based on preference
        """
        candidates = []

        # Parse the observation space
        inventory = observation['inventory'].astype(np.int)
        inventory_size = int(observation['inventory_size'][0])
        order = observation['ordered_piece'].astype(np.int)

        # Keep the original positions of pieces in inventory
        inventory_index = np.arange(len(inventory)).reshape(-1, 1)
        inventory = np.concatenate([inventory, inventory_index], axis=1)
        inventory = list(inventory)[:inventory_size]

        # Sort items by size (ascending) and pick the smallest that fits
        inventory.sort(key=lambda item: item[0] * item[1])
        rotated_order = order[::-1]
        for piece in inventory:
            if self._demand_fits_piece(order, piece):
                candidates.append(Cutting2DMazeAction(piece_id=piece[2], rotate=False, reverse_cutting_order=False))
            if self._demand_fits_piece(rotated_order, piece):
                candidates.append(Cutting2DMazeAction(piece_id=piece[2], rotate=True, reverse_cutting_order=False))

        return candidates

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy does not require the state() object to compute the action."""
        return False

    @override(Policy)
    def compute_top_action_candidates(self,
                                      observation: ObservationType,
                                      num_candidates: Optional[int],
                                      maze_state: Optional[MazeStateType] = None,
                                      env: Optional[BaseEnv] = None,
                                      actor_id: Optional[ActorID] = None,
                                      deterministic: bool = False) \
            -> Tuple[Sequence[ActionType], Sequence[float]]:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface
        """
        raise NotImplementedError

    @override(Policy)
    def compute_action(self, observation: ObservationType, maze_state: Optional[MazeStateType] = None,
                       env: Optional[BaseEnv] = None, actor_id: Optional[ActorID] = None,
                       deterministic: bool = False) -> ActionType:
        """implementation of :class:`~maze.core.agent.policy.Policy` interface
        """
        candidates = self.get_candidate_pieces(observation)

        if len(candidates) == 0:
            # No piece fits => this should not happen (we should always have the full-size raw piece in stock)
            raise ValueError("No piece in inventory fits the given order.")

        # convert MazeAction to agent action
        return self.action_conversion.maze_to_space(candidates[0])

    @staticmethod
    def _demand_fits_piece(current_demand: (int, int), piece: (int, int)) -> bool:
        """Whether the current_demand is smaller than piece, i.e. piece cut be cut to satisfy it."""
        return current_demand[0] <= piece[0] and current_demand[1] <= piece[1]
