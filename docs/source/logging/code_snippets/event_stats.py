class CuttingEvents(ABC):
    """Events related to the cutting process."""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def invalid_piece_selected(self):
        """An invalid piece is selected for cutting."""
