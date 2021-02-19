"""Event logs are written out through event log writer objects registered via the global event log writer registry."""

from abc import ABC, abstractmethod

from maze.core.log_events.episode_event_log import EpisodeEventLog


class LogEventsWriter(ABC):
    """Interface for modules writing out the event log data.

    Implement this interface for any custom event data logging.
    """

    @abstractmethod
    def write(self, episode_event_log: EpisodeEventLog) -> None:
        """
        Write out provided episode data (into a file, DB etc.)

        :param episode_event_log: Log of the episode events.
        """
