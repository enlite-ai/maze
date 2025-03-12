from copy import deepcopy
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from tutorial_maze_env.part06_struct_env.env.maze_env import maze_env_factory
from tutorial_maze_env.part06_struct_env.env.struct_env import StructuredCutting2DEnvironment
from maze.core.env.maze_env import MazeEnv


class MaskedStructuredCutting2DEnvironment(StructuredCutting2DEnvironment):
    """Structured environment version of the cutting 2D environment.
    The environment alternates between the two sub-steps:

    - Select cutting piece
    - Select cutting configuration (cutting order and cutting orientation)

    :param maze_env: The "flat" cutting 2D environment to wrap.
    """

    def __init__(self, maze_env: MazeEnv):
        super().__init__(maze_env)

        # add masks to observation spaces
        max_inventory = self.observation_conversion.max_pieces_in_inventory
        self._observation_spaces_dict[0].spaces["inventory_mask"] = \
            gym.spaces.Box(low=np.float32(0), high=np.float32(1), shape=(max_inventory,), dtype=np.float32)

        self._observation_spaces_dict[1].spaces["cutting_mask"] = \
            gym.spaces.Box(low=np.float32(0), high=np.float32(1), shape=(2,), dtype=np.float32)

    @staticmethod
    def _obs_selection_step(flat_obs: Dict[str, np.array]) -> Dict[str, np.array]:
        """Formats initial observation / observation available for the first sub-step."""
        observation = deepcopy(flat_obs)

        # prepare inventory mask
        sorted_order = np.sort(observation["ordered_piece"].flatten())
        sorted_inventory = np.sort(observation["inventory"], axis=1)

        observation["inventory_mask"] = np.all(observation["inventory"] > 0, axis=1).astype(np.float32)
        for i in np.nonzero(observation["inventory_mask"])[0]:
            # exclude pieces which do not fit
            observation["inventory_mask"][i] = np.all(sorted_order <= sorted_inventory[i])

        return observation

    @staticmethod
    def _obs_cutting_step(flat_obs: Dict[str, np.array], selected_piece_idx: int) -> Dict[str, np.array]:
        """Formats observation available for the second sub-step."""

        selected_piece = flat_obs["inventory"][selected_piece_idx]
        ordered_piece = flat_obs["ordered_piece"]

        # prepare cutting action mask
        cutting_mask = np.zeros((2,), dtype=np.float32)

        selected_piece = selected_piece.squeeze()
        if np.all(flat_obs["ordered_piece"] <= selected_piece):
            cutting_mask[0] = 1.0

        if np.all(flat_obs["ordered_piece"][::-1] <= selected_piece):
            cutting_mask[1] = 1.0

        return {"selected_piece": selected_piece,
                "ordered_piece": ordered_piece,
                "cutting_mask": cutting_mask}


def struct_env_factory(max_pieces_in_inventory: int, raw_piece_size: Tuple[int, int],
                       static_demand: List[Tuple[int, int]]) -> StructuredCutting2DEnvironment:
    """Convenience factory function that compiles a trainable structured environment.
    (for argument details see: Cutting2DEnvironment)
    """

    # init maze environment including observation and action interfaces
    env = maze_env_factory(max_pieces_in_inventory=max_pieces_in_inventory,
                           raw_piece_size=raw_piece_size,
                           static_demand=static_demand)

    # convert flat to structured environment
    return MaskedStructuredCutting2DEnvironment(env)
