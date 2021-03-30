"""Implementation of the basic event service for :obj:`~maze.core.env.core_env.CoreEnv` environments."""
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, TypeVar, Type, Set, Generator

from maze.core.events.event_record import EventRecord
from maze.core.events.event_topic_factory import event_topic_factory


class EventScope(ABC):
    """Base class for all services that integrate with the event system and therefore use EventService as their backend.

    Currently PubSub is the only concrete implementation."""

    @abstractmethod
    def notify_event(self, event: EventRecord) -> None:
        """
        Called on all event occurrences, if the respective event interface is registered for this scope.

        :param event: the event
        :return: None
        """

    @abstractmethod
    def clear_events(self) -> None:
        """
        Notification after the env step execution about the start of a new step.
        """


T = TypeVar('T')


class EventService:
    """Manages the recording of event invocations and provides simple event routing functionality.
    There is one EventService instance in every agent-environment loop, provided by the AgentEnvironmentContext.

    Within the environment the richer routing functionality provided by PubSub should be utilized, rather than
    directly interacting with this class.
    """

    class TopicInfo:
        """internal class to keep track of the topic state, including the collected events"""

        def __init__(self, interface_class: Type[T], scope: EventScope, proxy: T):
            self.interface_class = interface_class
            self.scope = scope
            self.events = deque()
            self.proxy = proxy

    def __init__(self):
        self.topics: Dict[T, EventService.TopicInfo] = dict()
        self.scopes: Set[EventScope] = set()

    def notify_event(self, event: EventRecord) -> None:
        """
        Notify the event service about a new event. This is invoked by the event topic proxies.

        :param event The event to be added.
        """
        topic = self.topics[event.interface_class]
        topic.events.append(event)

        if topic.scope:
            topic.scope.notify_event(event)

    def iterate_event_records(self) -> Generator[EventRecord, None, None]:
        """
        A generator to iterate all collected events
        """
        for topic_info in self.topics.values():
            for event in topic_info.events:
                yield event

    def create_event_topic(self, interface_class: Type[T], scope: EventScope = None) -> T:
        """
        Create a proxy instance of the event interface, which can be used conveniently to publish events.
        Returns an existing proxy, if it has been created before.

        :param interface_class: The class object of an abstract interface that defines the events as methods.
        :param scope: Every event topic can be bound to a single scope, e.g. a certain PubSub instance, to ensure
                      that all events of the topic `interface_class` will be received by this PubSub instance.
        :return: A proxy object, dynamically derived from the passed `interface_class`, that can be used to trigger
                 events.
        """
        # return topic proxy if it already exists
        topic = self.topics.get(interface_class)
        if topic:
            assert topic.scope == scope, 'same interface in different scopes ({} and {})'.format(topic.scope, scope)
            return topic.proxy

        recorder = event_topic_factory(interface_class, self.notify_event)

        topic = self.TopicInfo(interface_class, scope, recorder)
        self.topics[interface_class] = topic

        if scope is not None:
            self.scopes.add(scope)

        return recorder

    def clear_events(self) -> None:
        """
        Notify this service about the start of a new step. This should only be called by
        the AgentEnvironmentContext.

        Clears all collected events and notifies all registered scopes.
        """

        # clear events
        for topic in self.topics.values():
            topic.events.clear()

        # always clear the pubsub events to ensure that no events in the RewardAggregator are left
        self.clear_pubsub()

    def clear_pubsub(self) -> None:
        """ Clears the events collected by pubsub
        """
        for scope in self.scopes:
            scope.clear_events()
