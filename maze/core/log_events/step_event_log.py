"""StepEventLog keeps logs from one particular step. Belongs to episode event log."""

from typing import Iterable, Optional

from maze.core.events.event_collection import EventCollection
from maze.core.events.event_service import EventRecord


class StepEventLog:
    """Logs all events dispatched by the environment during one step.

    :param env_time: Internal time of the environment, if available. Otherwise Step ID.
    :param events: Events dispatched by an environment during one particular step.
    """

    def __init__(self, env_time: int, events: Optional[EventCollection] = None):
        self.events = events if events is not None else EventCollection()
        self.env_time = env_time

    def append(self, event: EventRecord):
        """Append a new event record to the step log."""
        self.events.append(event)

    def extend(self, event_list: Iterable[EventRecord]):
        """Append a list of events record to the step log."""
        self.events.extend(event_list)
