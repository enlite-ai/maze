class Cutting2DMazeState:
    """Cutting 2D environment MazeState representation.
    :param inventory: A list of pieces in inventory.
    :param max_pieces_in_inventory: Max number of pieces in inventory (inventory size).
    :param current_demand: Piece that should be produced in the next step.
    :param raw_piece_size: Size of a raw piece.
    """

    def __init__(self, inventory: [(int, int)], max_pieces_in_inventory: int,
                 current_demand: (int, int), raw_piece_size: (int, int)):
        self.inventory = inventory.copy()
        self.max_pieces_in_inventory = max_pieces_in_inventory
        self.current_demand = current_demand
        self.raw_piece_size = raw_piece_size
