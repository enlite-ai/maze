from abc import ABC
from typing import List, Type

import pytest

from maze.core.events.event_service import EventService
from maze.core.events.event_record import EventRecord
from maze.core.events.pubsub import Pubsub, Subscriber


class DummyInterface(ABC):
    def event1(self):
        raise NotImplementedError

    def event2(self, param1, param2):
        raise NotImplementedError


class DummySubscriber(Subscriber):
    def get_interfaces(self) -> List[Type[ABC]]:
        return [DummyInterface]


def test_pubsub_topic():
    pubsub = Pubsub(EventService())
    subscriber = DummySubscriber()

    publisher = pubsub.create_event_topic(DummyInterface)

    pubsub.register_subscriber(subscriber)

    publisher.event1()
    publisher.event2(param1=1, param2=2)

    assert len(subscriber.events) == 2

    record = subscriber.events[0]
    assert record.interface_method == DummyInterface.event1
    assert record.attributes == dict()

    record = subscriber.events[1]
    assert record.interface_method == DummyInterface.event2
    assert record.attributes == dict(param1=1, param2=2)


def test_pubsub_multiple_publishers():
    pubsub = Pubsub(EventService())
    subscriber = DummySubscriber()

    publisher1 = pubsub.create_event_topic(DummyInterface)
    publisher2 = pubsub.create_event_topic(DummyInterface)

    pubsub.register_subscriber(subscriber)

    publisher1.event1()
    publisher2.event2(param1=1, param2=2)

    assert len(subscriber.events) == 2

    record = subscriber.events[0]
    assert record.interface_method == DummyInterface.event1
    assert record.attributes == dict()

    record = subscriber.events[1]
    assert record.interface_method == DummyInterface.event2
    assert record.attributes == dict(param1=1, param2=2)


def test_subscriber_query_events():
    subscriber = DummySubscriber()

    ev1 = EventRecord(interface_class=DummyInterface, interface_method=DummyInterface.event1, attributes=dict())
    subscriber.notify_event(ev1)

    assert [ev1] == subscriber.query_events(event_spec=DummyInterface.event1)

    subscriber.reset()

    assert subscriber.query_events(event_spec=DummyInterface.event1) == []


def test_wrong_arguments():
    pubsub = Pubsub(EventService())
    topic = pubsub.create_event_topic(DummyInterface)

    with pytest.raises(TypeError):
        # noinspection PyArgumentList
        topic.event2(wrong_name=1)
