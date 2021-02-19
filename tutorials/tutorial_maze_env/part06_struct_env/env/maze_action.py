class Cutting2DMazeAction:
    """Environment cutting MazeAction object.
    :param piece_id: ID of the piece to cut.
    :param rotate: Whether to rotate the ordered piece.
    :param reverse_cutting_order: Whether to cut along Y axis first (not X first as normal).
    """

    def __init__(self, piece_id: int, rotate: bool, reverse_cutting_order: bool):
        self.piece_id = piece_id
        self.rotate = rotate
        self.reverse_cutting_order = reverse_cutting_order
