"""Implementation of a publisher-subscriber system.

The pubsub system provides means to decouple the reward calculation from the environment logic.
"""
import inspect
from abc import ABC, abstractmethod
from typing import Callable, List, Iterable, Dict, Union, Type, TypeVar

from maze.core.events.event_collection import EventCollection
from maze.core.events.event_service import EventScope, EventRecord, EventService


class Subscriber(ABC):
    """Event aggregation object.
    """

    def __init__(self):
        self.events = EventCollection()
        self.reset()

    @abstractmethod
    def get_interfaces(self) -> List[Type[ABC]]:
        """
        Specification of the event interfaces this subscriber wants to receive events from.
        Every subscriber must implement this configuration method.

        :return: A list of interface classes
        """

    def notify_event(self, event: EventRecord):
        """Notify the subscriber of a new event occurrence.

        :param event: the event
        :return: None
        """
        self.events.append(event)

    def reset(self):
        """Reset event aggregation.

        :return: None
        """
        self.events = EventCollection()

    def query_events(self, event_spec: Union[Callable, Iterable[Callable]]) -> Iterable:
        """Return all events collected at the current env step matching one or more given event types. The event
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

        for ev in event_spec:
            assert self._get_class_that_defined_method(ev) in self.get_interfaces(), \
                f"Event {ev} queried, but class not subscribed. Check your get_interfaces() implementation " \
                f"{self.get_interfaces()}."

        return self.events.query_events(event_spec)

    @staticmethod
    def _get_class_that_defined_method(method):
        """Source: https://stackoverflow.com/a/25959545"""
        if inspect.ismethod(method):
            for cls in inspect.getmro(method.__self__.__class__):
                if cls.__dict__.get(method.__name__) is method:
                    return cls
            method = method.__func__  # fallback to __qualname__ parsing
        if inspect.isfunction(method):
            cls = getattr(inspect.getmodule(method),
                          method.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
            if isinstance(cls, type):
                return cls
        return getattr(method, '__objclass__', None)  # handle special descriptor objects


# Generic type used for type hints in functions below.
T = TypeVar('T')


class Pubsub(EventScope):
    """Implementation of a message broker (Pubsub stands for publish and subscribe)."""

    def __init__(self, event_collector: EventService):
        super().__init__()

        self.event_collector = event_collector

        self.interface_to_subscribers: Dict[Type[T], List[Subscriber]] = dict()
        """map of interface class to the list of subscribed receivers"""

        self.subscribers: List[Subscriber] = list()
        """all registered subscribers"""

    def create_event_topic(self, interface_class: Type[T]) -> T:
        """
        Returns a proxy instance of the event interface, which the publisher can use to publish events. Behind the
        scenes every event invocation is serialized as EventRecord object and then routed to the registered
        subscribers.

        :param interface_class: The class object of an abstract interface that defines the events as methods.
        :return:    A proxy object, dynamically derived from the passed `interface_class`. This class is intended to
                    be used by the publisher to trigger events.
        """

        recorder = self.event_collector.create_event_topic(interface_class, self)

        # init list of registered subscribers if not already present
        if interface_class not in self.interface_to_subscribers:
            self.interface_to_subscribers[interface_class] = []

        return recorder

    def register_subscriber(self, new_subscriber: Subscriber):
        """ Register a subscriber to receive events from certain published interfaces,
            specified by Subscriber.get_interfaces()

        :param new_subscriber: the subscriber to be registered
        :return: None
        """
        self.subscribers.append(new_subscriber)

        # add the new subscriber to all interfaces for which events should be received
        for interface_class in new_subscriber.get_interfaces():
            if interface_class not in self.interface_to_subscribers:
                self.interface_to_subscribers[interface_class] = []

            registered_subscribers = self.interface_to_subscribers[interface_class]
            registered_subscribers.append(new_subscriber)

    def clear_events(self) -> None:
        """Resets the aggregated events of all registered subscribers

        :return: None
        """
        for subscriber in self.subscribers:
            subscriber.reset()

    def notify_event(self, event: EventRecord) -> None:
        """
        Notify about a new event. This is invoked by the EventService.

        :param event The event to be added.
        """
        for subscriber in self.interface_to_subscribers[event.interface_class]:
            subscriber.notify_event(event)
