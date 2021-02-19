from abc import ABC

import numpy as np
from maze.core.log_stats.event_decorators import define_step_stats, define_episode_stats, define_epoch_stats


class CuttingEvents(ABC):
    """Events related to the cutting process."""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def invalid_piece_selected(self):
        """An invalid piece is selected for cutting."""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def valid_cut(self, current_demand: (int, int), piece_to_cut: (int, int), raw_piece_size: (int, int),
                  cutting_area: float):
        """A valid cut was performed."""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def invalid_cut(self, current_demand: (int, int), piece_to_cut: (int, int), raw_piece_size: (int, int)):
        """Invalid cutting parameters have been speciefied."""


class InventoryEvents(ABC):
    """Events related to inventory management."""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def piece_discarded(self, piece: (int, int)):
        """The inventory is full and a piece has been discarded."""

    @define_epoch_stats(np.mean, input_name="step_mean", output_name="step_mean")
    @define_epoch_stats(max, input_name="step_max", output_name="step_max")
    @define_episode_stats(np.mean, output_name="step_mean")
    @define_episode_stats(max, output_name="step_max")
    @define_step_stats(None)
    def pieces_in_inventory(self, value: int):
        """Reports the count of pieces currently in the inventory."""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def piece_replenished(self):
        """A new raw cutting piece has been replenished."""
