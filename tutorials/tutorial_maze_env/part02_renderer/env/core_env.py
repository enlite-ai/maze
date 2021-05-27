from typing import Union, Tuple, Dict, Any

import numpy as np

from maze.core.env.core_env import CoreEnv
from maze.core.env.structured_env import ActorID

from .maze_state import Cutting2DMazeState
from .maze_action import Cutting2DMazeAction
from .inventory import Inventory
from .renderer import Cutting2DRenderer


class Cutting2DCoreEnvironment(CoreEnv):
    """Environment for cutting 2D pieces based on the customer demand. Works as follows:
     - Keeps inventory of 2D pieces available for cutting and fulfilling the demand.
     - Produces a new demand for one piece in every step (here a static demand).
     - The agent should decide which piece from inventory to cut (and how) to fulfill the given demand.
     - What remains from the cut piece is put back in inventory.
     - All the time, one raw (full-size) piece is available in inventory.
       (If it gets cut, it is replenished in the next step.)
     - Rewards are calculated to motivate the agent to consume as few raw pieces as possible.
     - If inventory gets full, the oldest pieces get discarded.

    :param max_pieces_in_inventory: Size of the inventory.
    :param raw_piece_size: Size of a fresh raw (= full-size) piece.
    :param static_demand: Order to issue in each step.
    """

    def __init__(self, max_pieces_in_inventory: int, raw_piece_size: (int, int),
                 static_demand: (int, int)):
        super().__init__()

        self.max_pieces_in_inventory = max_pieces_in_inventory
        self.raw_piece_size = tuple(raw_piece_size)
        self.current_demand = static_demand

        # initialize rendering
        self.renderer = Cutting2DRenderer()

        # setup environment
        self._setup_env()

    def _setup_env(self):
        """Setup environment."""
        self.inventory = Inventory(self.max_pieces_in_inventory, self.raw_piece_size)
        self.inventory.replenish_piece()

    def step(self, maze_action: Cutting2DMazeAction) \
            -> Tuple[Cutting2DMazeState, np.array, bool, Dict[Any, Any]]:
        """Summary of the step (simplified, not necessarily respecting the actual order in the code):
        1. Check if the selected piece to cut is valid (i.e. in inventory, large enough etc.)
        2. Attempt the cutting
        3. Replenish a fresh piece if needed and return an appropriate reward

        :param maze_action: Cutting MazeAction to take.
        :return: state, reward, done, info
        """

        info, reward = {}, 0
        replenishment_needed = False

        # check if valid piece id was selected
        if maze_action.piece_id >= self.inventory.size():
            info['error'] = 'piece_id_out_of_bounds'
        # perform cutting
        else:
            piece_to_cut = self.inventory.pieces[maze_action.piece_id]

            # attempt the cut
            if self.inventory.cut(maze_action, self.current_demand):
                info['msg'] = "valid_cut"
                replenishment_needed = piece_to_cut == self.raw_piece_size
            else:
                # assign a negative reward for invalid cutting attempts
                info['error'] = "invalid_cut"
                reward = -2

        # check if replenishment is required
        if replenishment_needed:
            self.inventory.replenish_piece()
            # assign negative reward if a piece has to be replenished
            reward = -1

        # compile env state
        maze_state = self.get_maze_state()

        return maze_state, reward, False, info

    def get_maze_state(self) -> Cutting2DMazeState:
        """Returns the current Cutting2DMazeState of the environment."""
        return Cutting2DMazeState(self.inventory.pieces, self.max_pieces_in_inventory,
                                  self.current_demand, self.raw_piece_size)

    def reset(self) -> Cutting2DMazeState:
        """Resets the environment to initial state."""
        self._setup_env()
        return self.get_maze_state()

    def close(self):
        """No additional cleanup necessary."""

    def seed(self, seed: int) -> None:
        """Seed random state of environment."""
        # No randomness in the env at this point
        pass

    def get_renderer(self) -> Cutting2DRenderer:
        """Cutting 2D renderer module."""
        return self.renderer

    # --- lets ignore everything below this line for now ---

    def get_serializable_components(self) -> Dict[str, Any]:
        pass

    def is_actor_done(self) -> bool:
        pass

    def actor_id(self) -> ActorID:
        pass

    def agent_counts_dict(self) -> Dict[Union[str, int], int]:
        pass
