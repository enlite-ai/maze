...

class Cutting2DCoreEnvironment(CoreEnv):
    """Environment for cutting 2D pieces based on the customer demand. Works as follows:
    ...
    :param reward_aggregator: Either an instantiated aggregator or a configuration dictionary.
    """

    def __init__(self, max_pieces_in_inventory: int, raw_piece_size: (int, int), static_demand: (int, int),
                 reward_aggregator: RewardAggregatorInterface):
        super().__init__()

        ...

        # init reward and register it with pubsub
        self.reward_aggregator = reward_aggregator
        self.pubsub.register_subscriber(self.reward_aggregator)

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

        # step execution finished, write step statistics
        self.inventory.log_step_statistics()

        # aggregate reward from events
        rewards = self.reward_aggregator.summarize_reward()

        # compile env state
        maze_state = self.get_maze_state()

        return maze_state, sum(rewards), False, info
