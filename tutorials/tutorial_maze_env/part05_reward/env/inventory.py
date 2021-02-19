from .events import InventoryEvents
from .maze_action import Cutting2DMazeAction


class Inventory:
    """Holds the inventory of 2D pieces and performs cutting.
    :param max_pieces_in_inventory: Size of the inventory. If full, the oldest pieces get discarded.
    :param raw_piece_size: Size of a fresh raw (= full-size) piece.
    :param inventory_events: Inventory event dispatch proxy.
    """

    def __init__(self, max_pieces_in_inventory: int, raw_piece_size: (int, int),
                 inventory_events: InventoryEvents):
        self.max_pieces_in_inventory = max_pieces_in_inventory
        self.raw_piece_size = raw_piece_size
        self.inventory_events = inventory_events

        self.pieces = []

    # == Inventory management ==

    def is_full(self) -> bool:
        """Checks weather all slots in the inventory are in use."""
        return len(self.pieces) == self.max_pieces_in_inventory

    def store_piece(self, piece: (int, int)) -> None:
        """Store the given piece.
        :param piece: Piece to store.
        """
        # If we would run out of storage space, discard the oldest piece first
        if self.is_full():
            self.pieces.pop(0)
            self.inventory_events.piece_discarded(piece=piece)

        self.pieces.append(piece)

    def replenish_piece(self) -> None:
        """Add a fresh raw piece to inventory."""
        self.store_piece(self.raw_piece_size)
        self.inventory_events.piece_replenished()

    # == Cutting ==

    def cut(self, maze_action: Cutting2DMazeAction, ordered_piece: (int, int)) -> bool:
        """Attempt to perform the cutting. Remains of the cut piece are put back to inventory.

        :param maze_action: the cutting maze_action to perform
        :param ordered_piece: Dimensions of the piece that we should produce
        :return True if the cutting was successful, False on error.
        """
        if maze_action.rotate:
            ordered_piece = ordered_piece[::-1]

        # Check the piece ID is valid
        if maze_action.piece_id >= len(self.pieces):
            return False

        # Check whether the cut is possible
        if any([ordered_piece[dim] > available_size for dim, available_size
                in enumerate(self.pieces[maze_action.piece_id])]):
            return False

        # Perform the cut
        cutting_order = [1, 0] if maze_action.reverse_cutting_order else [0, 1]
        piece_to_cut = list(self.pieces.pop(maze_action.piece_id))
        for dim in cutting_order:
            residual = piece_to_cut.copy()
            residual[dim] = piece_to_cut[dim] - ordered_piece[dim]
            piece_to_cut[dim] = ordered_piece[dim]
            if residual[dim] > 0:
                self.store_piece(tuple(residual))

        return True

    # == State representation ==

    def size(self) -> int:
        """Current size of the inventory."""
        return len(self.pieces)

    # == Step log ==

    def log_step_statistics(self):
        """Log inventory statistics once per step"""
        self.inventory_events.pieces_in_inventory(self.size())
