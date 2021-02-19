from .renderer import Cutting2DRenderer
...


class Cutting2DCoreEnvironment(CoreEnv):

    def __init__(self, max_pieces_in_inventory: int, raw_piece_size: (int, int), static_demand: (int, int)):
        super().__init__()

        # initialize rendering
        self.renderer = Cutting2DRenderer()
        ...

    def get_renderer(self) -> Cutting2DRenderer:
        """Cutting 2D renderer module."""
        return self.renderer
