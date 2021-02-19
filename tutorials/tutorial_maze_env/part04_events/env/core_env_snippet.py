...
from maze.core.events.pubsub import Pubsub
from .events import CuttingEvents, InventoryEvents
from .kpi_calculator import Cutting2dKpiCalculator


class Cutting2DCoreEnvironment(CoreEnv):

    def __init__(self, max_pieces_in_inventory: int, raw_piece_size: (int, int), static_demand: (int, int)):
        super().__init__()

        ...

        # init pubsub for event to reward routing
        self.pubsub = Pubsub(self.context.event_service)

        # KPIs calculation
        self.kpi_calculator = Cutting2dKpiCalculator()

    def _setup_env(self):
        """Setup environment."""
        inventory_events = self.pubsub.create_event_topic(InventoryEvents)
        self.inventory = Inventory(self.max_pieces_in_inventory, self.raw_piece_size, inventory_events)
        self.inventory.replenish_piece()

        self.cutting_events = self.pubsub.create_event_topic(CuttingEvents)

    def step(self, maze_action: Cutting2DMazeAction) -> Tuple[Cutting2DMazeState, np.array, bool, Dict[Any, Any]]:
        """Summary of the step (simplified, not necessarily respecting the actual order in the code):
        1. Check if the selected piece to cut is valid (i.e. in inventory, large enough etc.)
        2. Attempt the cutting
        3. Replenish a fresh piece if needed and return an appropriate reward

        :param maze_action: Cutting MazeAction to take.
        :return: maze_state, reward, done, info
        """

        info, reward = {}, 0
        replenishment_needed = False

        # check if valid piece id was selected
        if maze_action.piece_id >= self.inventory.size():
            self.cutting_events.invalid_piece_selected()
        # perform cutting
        else:
            piece_to_cut = self.inventory.pieces[maze_action.piece_id]

            # attempt the cut
            if self.inventory.cut(maze_action, self.current_demand):
                self.cutting_events.valid_cut(current_demand=self.current_demand, piece_to_cut=piece_to_cut,
                                              raw_piece_size=self.raw_piece_size)
                replenishment_needed = piece_to_cut == self.raw_piece_size
            else:
                # assign a negative reward for invalid cutting attempts
                self.cutting_events.invalid_cut(current_demand=self.current_demand, piece_to_cut=piece_to_cut,
                                                raw_piece_size=self.raw_piece_size)
                reward = -2

        # check if replenishment is required
        if replenishment_needed:
            self.inventory.replenish_piece()
            # assign negative reward if a piece has to be replenished
            reward = -1

        # step execution finished, write step statistics
        self.inventory.log_step_statistics()

        # compile env state
        maze_state = self.get_maze_state()

        return maze_state, reward, False, info

    def get_kpi_calculator(self) -> Cutting2dKpiCalculator:
        """KPIs are supported."""
        return self.kpi_calculator
