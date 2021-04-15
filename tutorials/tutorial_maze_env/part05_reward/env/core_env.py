from typing import Union, Tuple, Dict, Any

import numpy as np

from maze.core.env.core_env import CoreEnv
from maze.core.utils.seeding import set_random_states
from maze.core.events.pubsub import Pubsub

from .maze_state import Cutting2DMazeState
from .maze_action import Cutting2DMazeAction
from .inventory import Inventory
from .renderer import Cutting2DRenderer
from .events import CuttingEvents, InventoryEvents
from .kpi_calculator import Cutting2dKpiCalculator
from ..reward.default_reward import CuttingRewardAggregator


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
    :param reward_aggregator: Either an instantiated aggregator or a configuration dictionary.
    """

    def __init__(self, max_pieces_in_inventory: int, raw_piece_size: (int, int), static_demand: (int, int),
                 reward_aggregator: CuttingRewardAggregator):
        super().__init__()

        self.max_pieces_in_inventory = max_pieces_in_inventory
        self.raw_piece_size = tuple(raw_piece_size)
        self.current_demand = static_demand

        # initialize rendering
        self.renderer = Cutting2DRenderer()

        # init pubsub for event to reward routing
        self.pubsub = Pubsub(self.context.event_service)

        # KPIs calculation
        self.kpi_calculator = Cutting2dKpiCalculator()

        # setup environment
        self._setup_env()

        # init reward and register it with pubsub
        self.reward_aggregator = reward_aggregator
        self.pubsub.register_subscriber(self.reward_aggregator)

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

        :param maze_action: Cutting maze_action to take.
        :return: state, reward, done, info
        """

        info = {}
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

        # check if replenishment is required
        if replenishment_needed:
            self.inventory.replenish_piece()
            # assign negative reward if a piece has to be replenished

        # step maze_action finished, write step statistics
        self.inventory.log_step_statistics()

        # aggregate reward from events
        reward = self.reward_aggregator.collect_rewards()

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
        set_random_states(seed)

    def get_renderer(self) -> Cutting2DRenderer:
        """Cutting 2D renderer module."""
        return self.renderer

    def is_actor_done(self) -> bool:
        """Returns True if the just stepped actor is done, which is different to the done flag of the environment."""
        return False

    def actor_id(self) -> Tuple[Union[str, int], int]:
        """Returns the currently executed actor along with the policy id. The id is unique only with
        respect to the policies (every policy has its own actor 0).
        Note that identities of done actors can not be reused in the same rollout.

        :return: The current actor, as tuple (policy id, actor number).
        """
        return 0, 0

    @property
    def agent_counts_dict(self) -> Dict[Union[str, int], int]:
        """Returns the count of agents for individual sub-steps (or -1 for dynamic agent count).

        As this is a single-step single-agent environment, in which 1 agent gets to act during sub-step 0,
        we return {0: 1}.
        """
        return {0: 1}

    def get_kpi_calculator(self) -> Cutting2dKpiCalculator:
        """KPIs are supported."""
        return self.kpi_calculator

    # --- lets ignore everything below this line for now ---

    def get_serializable_components(self) -> Dict[str, Any]:
        pass
