"""Implements a collection of EventRecord instances to have a stable interface regardless of
   potential data structure optimizations."""
from collections import deque
from typing import Callable, Iterable, MutableSequence, Union, List

from maze.core.events.event_service import EventRecord


class EventCollection:
    """A collection of EventRecord instances that can be queried by event specification."""
    def __init__(self, events: Iterable[EventRecord] = ()):
        self.events: MutableSequence[EventRecord] = deque(events)

    def append(self, event: EventRecord):
        """Append a new event record to the collection."""
        self.events.append(event)

    def extend(self, event_list: Iterable[EventRecord]):
        """Extends self.events with a list of new event records."""
        self.events.extend(event_list)

    def query_events(self, event_spec: Union[Callable, Iterable[Callable]]) -> Iterable:
        """ Return all events collected at the current env step matching one or more given event types. The event
            types are specified by the interface member function object itself.

            Event calls are recorded as EventRecord, an object providing access to the passed arguments of the event
            method.

        :param event_spec:  Specifies the event type by the interface member function. Can either be a single event
                            type specification or a list of specifications.
        :return:            An iterable to the event objects.
        """

        # normalize event specification, convert to a list with a single item if necessary
        if not isinstance(event_spec, Iterable):
            event_spec = [event_spec]

        return [event for event in self.events if event.interface_method in event_spec]

    def __len__(self):
        """Support len() operator."""
        return len(self.events)

    def __getitem__(self, item):
        """Support subscription [] operator"""
        return self.events[item]
