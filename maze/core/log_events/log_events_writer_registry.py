"""
Event log writers need to be globally registered in order to receive episode event logs streamed from
environments.
"""

from typing import List

from maze.core.log_events.episode_event_log import EpisodeEventLog
from maze.core.log_events.log_events_writer import LogEventsWriter


class LogEventsWriterRegistry:
    """Handles registration of event log writers.

    Registered writers will be forwarded episode event log data at the end of each episode.
    """

    # List of registered writers
    writers: List[LogEventsWriter] = []

    @classmethod
    def register_writer(cls, writer: LogEventsWriter) -> None:
        """Register a writer. Each writer will receive all globally recorded event logs.

        :param writer: Event log writer to register.
        """
        cls.writers.append(writer)

    @classmethod
    def record_event_logs(cls, episode_event_log: EpisodeEventLog) -> None:
        """
        Write event log data through all registered event log writers.

        :param episode_event_log: Log of recorded environment events.
        """
        for writer in cls.writers:
            writer.write(episode_event_log)
