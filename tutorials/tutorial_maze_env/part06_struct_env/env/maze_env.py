from typing import List, Tuple

from maze.core.env.core_env import CoreEnv
from maze.core.env.maze_env import MazeEnv
from maze.core.env.action_conversion import ActionConversionInterface
from maze.core.env.observation_conversion import ObservationConversionInterface

from .core_env import Cutting2DCoreEnvironment
from ..reward.default_reward import DefaultRewardAggregator
from ..space_interfaces.dict_observation_conversion import ObservationConversion
from ..space_interfaces.dict_action_conversion import ActionConversion


class Cutting2DEnvironment(MazeEnv[Cutting2DCoreEnvironment]):
    """Maze environment for 2d cutting.

    :param core_env: The underlying core environment.
    :param action_conversion: An action conversion interface.
    :param observation_conversion: An observation conversion interface.
    """

    def __init__(self,
                 core_env: CoreEnv,
                 action_conversion: ActionConversionInterface,
                 observation_conversion: ObservationConversionInterface):
        super().__init__(core_env=core_env,
                         action_conversion_dict={0: action_conversion},
                         observation_conversion_dict={0: observation_conversion})


def maze_env_factory(max_pieces_in_inventory: int, raw_piece_size: Tuple[int, int],
                     static_demand: List[Tuple[int, int]]) -> Cutting2DEnvironment:
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
