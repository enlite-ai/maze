"""Internal, functionality used by the event system to construct proxies for event recording."""
import inspect
from typing import TypeVar, Type, Callable

from maze.core.events.event_record import EventRecord

T = TypeVar('T')


def event_topic_factory(interface_class: Type[T], fn_notify_event: Callable[[EventRecord], None]) -> T:
    """Constructs a proxy instance of the event interface, as required by EventService and LogStatsAggregator.

    :param interface_class: The class object of an abstract interface that defines the events as methods.
    :param fn_notify_event: The proxy forwards all method invocations to fn_notify_event
    :return: A proxy object, dynamically derived from the passed `interface_class`.
    """

    # dynamically create a derived proxy class
    proxy_name = interface_class.__name__ + "Proxy"
    proxy_class = type(proxy_name, (interface_class,), dict())
    proxy = proxy_class()

    # implement the interface methods
    for name in dir(interface_class):
        # skip hidden attributes
        if name.startswith('_'):
            continue

        attr = getattr(interface_class, name)

        # only interested in class methods
        if not callable(attr):
            continue

        setattr(proxy, name, _recorder_factory(interface_class, interface_method=attr, fn_notify_event=fn_notify_event))

    return proxy


def _recorder_factory(interface_class, interface_method, fn_notify_event):
    """
    internal method, returns the method used to record the events
    """
    signature = inspect.signature(interface_method)
    event_attribute_names = [p.name for p in list(signature.parameters.values())[1:]]

    def _record(*args, **kwargs) -> None:
        if len(args) > 1:
            raise TypeError('event methods with more than one argument must be called with named arguments')

        if len(args) == 1:
            if len(event_attribute_names) > 1:
                raise TypeError('event methods with more than one argument must be called with named arguments')

            value = args[0]
            attribute_name = event_attribute_names[0]
            fn_notify_event(EventRecord(interface_class, interface_method, attributes={attribute_name: value}))
            return

        # call the original interface, to raise error on signature mismatch
        for key in kwargs.keys():
            if key not in event_attribute_names:
                raise TypeError("got an unexpected keyword argument {}".format(key))

        fn_notify_event(EventRecord(interface_class, interface_method, attributes=kwargs))

    return _record

