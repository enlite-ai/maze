from typing import Dict
from gymnasium import spaces
from maze.core.env.action_conversion import ActionConversionInterface

from ..env.maze_action import Cutting2DMazeAction
from ..env.maze_state import Cutting2DMazeState


class ActionConversion(ActionConversionInterface):
    """Converts agent actions to actual environment MazeActions.

    :param max_pieces_in_inventory: Size of the inventory
    """

    def __init__(self, max_pieces_in_inventory: int):
        self.max_pieces_in_inventory = max_pieces_in_inventory

    def space_to_maze(self, action: Dict[str, int], maze_state: Cutting2DMazeState) -> Cutting2DMazeAction:
        """Converts agent dictionary action to environment MazeAction object."""
        return Cutting2DMazeAction(piece_id=action["piece_idx"],
                                  rotate=bool(action["cut_rotation"]),
                                  reverse_cutting_order=bool(action["cut_order"]))

    def maze_to_space(self, maze_action: Cutting2DMazeAction) -> Dict[str, int]:
        """Converts environment MazeAction object to agent dictionary action."""
        return {"piece_idx": maze_action.piece_id,
                "cut_rotation": int(maze_action.rotate),
                "cut_order": int(maze_action.reverse_cutting_order)}

    def space(self) -> spaces.Dict:
        """Returns Gym dict action space."""
        return spaces.Dict({
            "piece_idx": spaces.Discrete(self.max_pieces_in_inventory),  # Which piece should be cut
            "cut_rotation": spaces.Discrete(2),  # Rotate: (yes / no)
            "cut_order": spaces.Discrete(2)      # Cutting order: (xy / yx)
        })
