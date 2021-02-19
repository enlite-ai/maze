"""Episode event log is the main unit of logging environment events in granular form."""

from typing import Iterable, Union, Callable
from typing import List

from maze.core.log_events.step_event_log import StepEventLog


class EpisodeEventLog:
    """Keeps logs of all events dispatched by an environment during one episode.

    :param episode_id: ID of the episode the events belong to
    """

    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.step_event_logs: List[StepEventLog] = []

    def query_events(self, event_spec: Union[Callable, Iterable[Callable]]) -> Iterable:
        """Query events across the whole episode.

        :param event_spec: Specification of events to query
        :return: List of events from this episode that
        """
        event_records = []
        for step_log in self.step_event_logs:
            event_records += step_log.events.query_events(event_spec)
        return event_records
