...


def maze_env_factory(max_pieces_in_inventory: int, raw_piece_size: (int, int),
                     static_demand: (int, int)) -> Cutting2DEnvironment:
    """Convenience factory function that compiles a trainable maze environment.
    (for argument details see: Cutting2DCoreEnvironment)
    """

    # init reward aggregator
    reward_aggregator = DefaultRewardAggregator(invalid_action_penalty=-2, raw_piece_usage_penalty=-1)

    # init core environment
    core_env = Cutting2DCoreEnvironment(max_pieces_in_inventory=max_pieces_in_inventory,
                                        raw_piece_size=raw_piece_size,
                                        static_demand=static_demand,
                                        reward_aggregator=reward_aggregator)

    # init maze environment including observation and action interfaces
    action_conversion = ActionConversion(max_pieces_in_inventory=max_pieces_in_inventory)
    observation_conversion = ObservationConversion(raw_piece_size=raw_piece_size,
                                                   max_pieces_in_inventory=max_pieces_in_inventory)
    return Cutting2DEnvironment(core_env, action_conversion, observation_conversion)
