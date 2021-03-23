from maze.core.log_stats.log_stats import LogStatsConsumer, LogStats, LogStatsLevel
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper


class SinkHoleConsumer(LogStatsConsumer):
    """Sink hole statistics consumer. Discards all statistics on receive."""

    def receive(self, stat: LogStats):
        """Do not keep the received statistics."""
        pass


def disable_epoch_level_stats(env: LogStatsWrapper):
    """Disable collection of statistics on epoch level to save memory.

    This is useful in distributed vectorized env scenarios as episode statistics are shipped to the main worker/node
    and there is no need to keep them and aggregate them on worker level.
    """
    sink_hole_consumer = SinkHoleConsumer()
    env.stats_map[LogStatsLevel.EPOCH] = sink_hole_consumer
    env.stats_map[LogStatsLevel.EPISODE].consumers = [sink_hole_consumer]
    return env
