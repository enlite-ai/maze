...
from .events import InventoryEvents


class Inventory:
    """Holds the inventory of 2D pieces and performs cutting.
    :param max_pieces_in_inventory: Size of the inventory. If full, the oldest pieces get discarded.
    :param raw_piece_size: Size of a fresh raw (= full-size) piece.
    :param inventory_events: Inventory event dispatch proxy.
    """

    def __init__(self, max_pieces_in_inventory: int, raw_piece_size: (int, int),
                 inventory_events: InventoryEvents):
        ...

        self.inventory_events = inventory_events

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

    def log_step_statistics(self):
        """Log inventory statistics once per step"""
        self.inventory_events.pieces_in_inventory(self.size())
