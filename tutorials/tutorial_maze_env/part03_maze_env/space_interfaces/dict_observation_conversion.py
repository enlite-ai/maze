import numpy as np
from typing import Dict
from gymnasium import spaces

from maze.core.annotations import override
from maze.core.env.observation_conversion import ObservationConversionInterface
from ..env.maze_state import Cutting2DMazeState


class ObservationConversion(ObservationConversionInterface):
    """Cutting 2d environment state to dictionary observation.

    :param max_pieces_in_inventory: Size of the inventory. If inventory gets full, the oldest pieces get discarded.
    :param raw_piece_size: Size of a fresh raw (= full-size) piece
    """

    def __init__(self, raw_piece_size: (int, int), max_pieces_in_inventory: int):
        self.max_pieces_in_inventory = max_pieces_in_inventory
        self.raw_piece_size = raw_piece_size

    @override(ObservationConversionInterface)
    def maze_to_space(self, maze_state: Cutting2DMazeState) -> Dict[str, np.ndarray]:
        """Converts core environment state to a machine readable agent observation."""

        # Convert inventory to numpy array and stretch it to full size (filling with zeros)
        inventory_state = maze_state.inventory
        inventory_state += [(0, 0)] * (self.max_pieces_in_inventory - len(maze_state.inventory))

        # Compile dict space observation
        return {'inventory': np.asarray(inventory_state, dtype=np.float32),
                'inventory_size': np.asarray([len(maze_state.inventory)], dtype=np.float32),
                'ordered_piece': np.asarray(maze_state.current_demand, dtype=np.float32)}

    @override(ObservationConversionInterface)
    def space_to_maze(self, observation: Dict[str, np.ndarray]) -> Cutting2DMazeState:
        """Converts agent observation to core environment state (not required for this example)."""
        raise NotImplementedError

    @override(ObservationConversionInterface)
    def space(self) -> spaces.Dict:
        """Return the Gym dict observation space based on the given params.

        :return: Gym space object
            - inventory: max_pieces_in_inventory x 2 (x/y-dimensions of pieces in inventory)
            - inventory_size: scalar number of pieces in inventory
            - ordered_piece: 2d vector holding x/y-dimension of customer ordered piece
        """
        return spaces.Dict({
            'inventory': spaces.Box(low=np.zeros((self.max_pieces_in_inventory, 2), dtype=np.float32),
                                    high=np.vstack([[self.raw_piece_size[0] + 1, self.raw_piece_size[1] + 1]] *
                                                   self.max_pieces_in_inventory).astype(np.float32),
                                    dtype=np.float32),
            'inventory_size': spaces.Box(low=np.float32(0), high=self.max_pieces_in_inventory + 1,
                                         shape=(1,), dtype=np.float32),
            'ordered_piece': spaces.Box(low=np.float32(0), high=np.float32(max(self.raw_piece_size) + 1),
                                         shape=(2,), dtype=np.float32)
        })
