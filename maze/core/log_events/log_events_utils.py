"""Simple default setup for event logging into CSV files."""

from maze.core.log_events.log_events_writer_registry import LogEventsWriterRegistry
from maze.core.log_events.log_events_writer_tsv import LogEventsWriterTSV
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper


class SimpleEventLoggingSetup:
    """Simple setup for logging of environment events with all their attributes.

    Events will be logged into CSV files in "event_logs" directory.

    :param env: Env to log events from (needs to be wrapper in LogEventsWrapper)
    """

    def __init__(self, env: LogStatsWrapper):
        self.env = env

    def __enter__(self) -> None:
        """Register event log writer."""
        LogEventsWriterRegistry.register_writer(LogEventsWriterTSV())

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Ensure the event logs are written out in case the rollout ended in the middle of episode
        (without env reset at the end).
        """
        self.env.write_epoch_stats()
