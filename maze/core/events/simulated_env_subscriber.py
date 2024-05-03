"""File holding subscriber for simulated envs to share the events with the main env."""
from __future__ import annotations

from typing import Callable

from maze.core.agent.policy import Policy
from maze.core.events.event_collection import EventCollection
from maze.core.events.event_record import EventRecord
from maze.core.events.pubsub import Subscriber


def share_sim_env_events_to_main_env(func_obj: Callable) -> Callable:
    """Decorator, indicating sharing events from simulated env to main env.

    :param func_obj: The actual function to be decorated.
    :return: The decorated function.
    """

    def decorator(*args, **kwargs):
        # Check for policy
        assert len(args) == 1
        assert isinstance(args[0], Policy)

        # Get policy
        policy = args[0]

        # Check for subscriber
        shared_event_subscriber = getattr(policy, 'shared_event_subscriber', SimulationEnvSharedEventsSubscriber())
        if not hasattr(policy, 'shared_event_subscriber'):
            # Add Subscriber
            setattr(policy, 'shared_event_subscriber', shared_event_subscriber)

            # Get main env
            env = None
            for key, val in kwargs.items():
                if key == 'env':
                    env = val
                    break
            assert env is not None

            # Get available event topics/interfaces from main env and register the subscriber
            # Note: Simulated envs are replicates of main env, therefore they have the same events
            for interface_cls, interface_ref in env.core_env.context.event_service.topics.items():
                policy.shared_event_subscriber.set_interface(interface_cls, interface_ref)
            for pubsub in policy.sim_env.core_env.context.event_service.scopes:
                pubsub.register_subscriber(policy.shared_event_subscriber)
        else:
            assert isinstance(shared_event_subscriber, SimulationEnvSharedEventsSubscriber)

        # Call decorated function
        func_return = func_obj(*args, **kwargs)

        # Publish subscribed events from simulated env to main env and clear subscriber
        policy.shared_event_subscriber.publish_events()
        policy.shared_event_subscriber.clear()

        return func_return

    return decorator


class SimulationEnvSharedEventsSubscriber(Subscriber):
    """Subscriber for simulated envs to share the events with the main env.

    The subscriber subscribes to the pubsub object of the simulated env in order to receive events. The events are
    forwarded to the main env.
    """
    def __init__(self):
        super().__init__()

        # Interfaces this subscriber listens to
        self.interface_classes = []
        self.interface_references = []

        # Record events
        self.subscribed_events_records = EventCollection()

    def publish_events(self) -> None:
        """Publish events to the main environment."""
        for event in self.subscribed_events_records:
            # Find the corresponding event interface class and reference
            for cls_idx, interface_class in enumerate(self.interface_classes):
                if event.interface_class == interface_class:
                    # Get event reference proxy and publish event to main environment
                    ref_proxy = self.interface_references[cls_idx].proxy
                    rec_func = getattr(ref_proxy, event.interface_method.__name__, None)
                    assert rec_func is not None, f'{ref_proxy} has no method {event.interface_method.__name__}'
                    assert len(event.attributes) == 1, 'Only event methods with one argument are supported'
                    for value in event.attributes.values():
                        # Publish
                        rec_func(value)

    def notify_event(self, event: EventRecord):
        """Notify the subscriber of a new event occurrence.

        Note: This subscriber only listens to events marked with the shared_event decorator.

        :param event: the event
        :return: None
        """
        if getattr(event.interface_method, 'shared_event', False):
            self.subscribed_events_records.append(event)

    def set_interface(self, interface_class: any, interface_reference: any) -> None:
        """Set interfaces to listen to that contain events that are to be shared.

        :param interface_class: Class of interface.
        :param interface_reference: Reference of interface.
        """
        # Safety check
        assert interface_class == interface_reference.interface_class

        # Get and filter class methods
        interface_methods = [
            attr
            for attr in dir(interface_class) if callable(getattr(interface_class, attr)) and not attr.startswith('__')
        ]

        # Check if an event should be shared
        has_shared_events = False
        for interface_method in interface_methods:
            interface_method_ = getattr(interface_class, interface_method, None)
            has_shared_events = getattr(interface_method_, 'shared_event', False)
            if has_shared_events:
                break

        # Register interfaces with shared events
        if has_shared_events:
            self.interface_classes.append(interface_class)
            self.interface_references.append(interface_reference)

    def get_interfaces(self) -> list:
        """Returns interfaces to listen to.

        :return: Interfaces to listen to.
        """
        return self.interface_classes

    def clear(self) -> None:
        """Reset event aggregation."""
        self.subscribed_events_records = EventCollection()
