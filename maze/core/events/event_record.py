"""Internal auxiliary class"""
from abc import ABC
from typing import Callable, Type


class EventRecord(object):
    """This auxiliary class is used to record calls to the event interface
    """

    def __init__(self, interface_class: Type[ABC], interface_method: Callable, attributes: dict):
        """
        Constructor

        :param interface_method: The event is identified by its interface member function object
        :param attributes: kwargs dictionary of the event invocation
        """
        self.interface_class = interface_class
        self.interface_method = interface_method
        self.attributes = attributes

    def __getattr__(self, argument_name):
        """
        Make event parameters conveniently available via 'event.argument_name'
        (shorthand to event.attributes['argument_name'])

        :param argument_name: event argument name
        :return: event argument value for the given argument_name
        """

        # this is required for deep copying the env context e.g., when calling clone_from on a CoreEnv.
        if argument_name == "__deepcopy__":
            return None

        return self.attributes[argument_name]

    # Pickle support
    # --------------
    # Methods below need to be overriden because pickle attempts to call them first (before the default
    # methods), and they get caught by the __getattr__ accessor above otherwise.

    def __getstate__(self):
        """Enable pickling using the standard __dict__ object.

        (Needed for standard pickling behavior because of the __getattr__ override above.)"""
        return self.__dict__

    def __setstate__(self, state):
        """Enable unpickling using the standard __dict__ update.

        (Needed for standard unpickling behavior because of the __getattr__ override above.)"""
        self.__dict__.update(state)
