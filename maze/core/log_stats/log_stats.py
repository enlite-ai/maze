"""Statistics Logging"""
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Union, Callable, Dict, Optional, NamedTuple, List, Type, TypeVar, Tuple, Any

import numpy as np
from maze.core.events.event_record import EventRecord
from maze.core.events.event_topic_factory import event_topic_factory

# define the LogStats type
LogStatsValue = Union[int, float, np.ndarray, dict]
"""Basic data structure for log statistics"""

LogStatsGroup = Tuple[Union[str, int], ...]
"""Basic data structure for log statistics"""

LogStatsKey = NamedTuple("LogStatsKey",
                         [
                             ("event", Callable),
                             ("output_name", str),
                             ("group", Optional[LogStatsGroup])
                         ])
"""Basic data structure for log statistics"""

LogStats = Dict[LogStatsKey, LogStatsValue]
"""Basic data structure for log statistics"""


class LogStatsConsumer(ABC):
    """
    An interface to receive log statistics. Implemented by the LogStatAggregator class to receive logs
    from the subjacent aggregator.
    """

    @abstractmethod
    def receive(self, stat: LogStats):
        """
        Receive a statistics object from an aggregator. This might be called multiple times in case we consume
        statistics from multiple LogStatAggregator objects.

        :param stat: The statistics dictionary
        """


class LogStatsLevel(Enum):
    """
    Log statistics aggregation levels.
    """

    STEP = 1
    """aggregator receives individual events and produces step statistics"""

    EPISODE = 2
    """aggregator receives step statistics and produces episode statistics"""

    EPOCH = 3
    """aggregator receives episode statistics and produces epoch statistics"""


class LogStatsAggregator(LogStatsConsumer):
    """
    Complements the event system by providing aggregation functionality. How the events are aggregated
    is specified by the event interface decorators (see event_decorators.py).

    Note that the statistics calculation for episode aggregators will automatically be
    triggered on every `increment_log_step()` call.
    """

    def __init__(self, level: LogStatsLevel, *consumers: LogStatsConsumer):
        """
        Constructor

        :param level: The aggregation level of this object, supported values:
                      LogStatsLevel.STEP, LogStatsLevel.EPISODE, LogStatsLevel.EPOCH.
        :param consumers: Optionally an arbitrary number of consumers can be passed (a consumer being either
                          the downstream aggregator or a log writer)
        """
        self.level = level
        self.consumers = list(consumers)

        self.input: Dict[LogStatsKey, list] = defaultdict(list)

        self.last_stats: Optional[LogStats] = None
        """keep track of the previous statistics, required e.g. for cumulative statistics"""

        self.last_stats_step: Optional[int] = None
        """step number of the last statistics calculation"""

        self.cumulative_stats = dict()
        """keep track of all cumulative stats, required e.g. for an epoch where the previous epoch did not finish any 
        episode"""

        if level == LogStatsLevel.EPOCH:
            GlobalLogState.hook_on_log_step.append(self._hook_on_log_step)

    def receive(self, stats: LogStats) -> None:
        """
        Receive statistics from the subjacent aggregation level

        :param stats: The statistics dictionary
        """
        # add the individual statistics values
        for (event, name, group), value in stats.items():
            self.add_value(event, value, name=name, group=group)

    def clear_inputs(self) -> None:
        """Clear the input statistics (and start fresh)."""
        self.input: Dict[LogStatsKey, list] = defaultdict(list)

    def add_event(self, event_record: EventRecord) -> None:
        """
        Add a recorded event to this aggregator.

        The aggregator only keeps track of event/attributes with relevant statistics decoration, everything else is
        filtered out immediately.

        :param event_record:
        """
        # find out if there is an aggregation decorator specified for this event
        event = event_record.interface_method
        input_to_reducers = getattr(event, self.level.name, None)

        # no aggregation defined on this event
        if not input_to_reducers:
            return

        # handle grouping
        group_by_list = getattr(event, "group_by", None)
        group = tuple(event_record.attributes[group_by] for group_by in group_by_list) if group_by_list else None

        # add all event attributes if there are reducers operating on the event level
        if None in input_to_reducers:
            # flatten the dict if there is only a single event attribute
            self.add_value(event,
                           self._flatten_single_attribute(event_record.attributes, group_by_list),
                           name=None,
                           group=group)

        # iterate over the individual event attributes
        for attribute_name, attribute_value in event_record.attributes.items():
            reducers = input_to_reducers.get(attribute_name, None)
            if not reducers:
                continue

            self.add_value(event, attribute_value, name=attribute_name, group=group)

    def add_value(self,
                  event: Callable,
                  value: LogStatsValue,
                  name: str = None,
                  group: LogStatsGroup = None) -> None:
        """
        Add a single value to this aggregator.

        :param event: The event interface method the given value belongs to
        :param value: The value to add
        :param name: There may be multiple statistics per event, the name is used to refer to a specific statistics
                     built from the event records
        :param group: The group identifier if this event is grouped
        """
        # append the new value to the list
        self.input[(event, name, group)].append(value)

    def register_consumer(self, consumer: LogStatsConsumer) -> None:
        """
        Register a new consumer to receive the readily calculated statistics of this aggregator.

        :param consumer: The consumer to add
        """
        self.consumers.append(consumer)

    def _hook_on_log_step(self) -> None:
        """This method is invoked on incrementing the global log step."""
        # skip auto-reduce if a manual update already took place
        if GlobalLogState.global_step == self.last_stats_step:
            return

        self.reduce()

    def reduce(self) -> LogStats:
        """
        Consume the aggregated values by
        - calculating the statistics
        - sending the statistics to the consumers
        - resetting the values for the next aggregation step

        :return Returns the statistics object. The same object has been sent to the consumers.
        """
        if self.level == LogStatsLevel.EPOCH:
            assert self.last_stats_step != GlobalLogState.global_step, "reduce() called twice on same global log step"

        all_values = defaultdict(list)

        # first stage: collect all values and handle group projection
        for (event, input_name, group_tuple), values in self.input.items():
            # retrieve the aggregation info for this level (created by the decorators)
            input_to_reducers = getattr(event, self.level.name, None)

            if not input_to_reducers or input_name not in input_to_reducers:
                continue

            all_grouping_attributes = getattr(event, "group_by", None)

            # iterate the aggregations
            reducers = input_to_reducers[input_name]
            for reduce_function, output_name, grouping_attribute, cumulative in reducers:
                projected_group = self._project_group(group_tuple, grouping_attribute, all_grouping_attributes, event)

                all_values[(event, output_name, projected_group, cumulative, reduce_function)].extend(values)

        # second stage: execute reducers
        aggregated_stats = dict()

        for (event, output_name, group_tuple, cumulative, reduce_function), values in all_values.items():
            try:
                stats_key = (event, output_name, group_tuple)
                reduced_value = self._reduce(reduce_function, values, event)

                # handle cumulative statistics: add value from previous run
                if cumulative:
                    if stats_key not in self.cumulative_stats:
                        self.cumulative_stats[stats_key] = reduced_value
                    else:
                        self.cumulative_stats[stats_key] += reduced_value
                    continue

                aggregated_stats[stats_key] = reduced_value
            except AssertionError:
                # keep assertions as is
                raise
            except Exception as e:
                # wrap the exception to make it easier to trace the event that caused the exception
                raise ValueError(f"failed to reduce event {event}") from e

        aggregated_stats.update(self.cumulative_stats)

        # send the statistics to the consumers
        for consumer in self.consumers:
            consumer.receive(aggregated_stats)

        # reset the collected statistics
        self.input.clear()
        self.last_stats = aggregated_stats

        # keep track of the global step to detect unintended reduce steps
        self.last_stats_step = GlobalLogState.global_step

        return aggregated_stats

    @classmethod
    def _reduce(cls, reduce_function: Callable, values: List[Any], event: Callable) -> Any:
        """Execute the given reduce function.

        Note that the input and output of the reduce function is not limited to a (lists of) attribute dictionaries
        as defined in :attr:`maze.core.events.event_record.EventRecord.attributes <EventRecord.attributes>`,
        but can be any custom data structure.
        """
        # special case: no aggregation
        if reduce_function is None:
            if len(values) > 1:
                raise AssertionError("event with aggregation skip operation 'None' is dispatched "
                                     "more than once per step: {} ".format(event))

            # get the first and only value
            first_value = values[0]
            if type(first_value) is not dict:
                return first_value

            # if there is more than one attribute defined for the event, return the entire dict
            if len(first_value) > 1:
                return tuple(first_value.values())

            return next(iter(first_value.values()))

        # execute the actual aggregation operation
        return reduce_function(values)

    @classmethod
    def _flatten_single_attribute(cls,
                                  attributes: Dict[str, Any],
                                  group_by_list: Optional[List[str]]
                                  ) -> Union[Dict[str, Any], Any]:
        """Unwrap single attributes (events with just one value) from the dictionary. Return the dict as is if it
        contains more than one attribute."""
        if group_by_list:
            attributes_minus_group = attributes.copy()
            for group_by in group_by_list:
                attributes_minus_group.pop(group_by)

            return cls._flatten_single_attribute(attributes_minus_group, None)

        if len(attributes) == 1:
            return next(iter(attributes.values()))

        return attributes

    @classmethod
    def _project_group(cls,
                       group_tuple: LogStatsGroup,
                       group_attribute: str,
                       all_group_attributes: List[str],
                       event: Callable) -> LogStatsGroup:
        """"helper to project groups, ie. setting all other tuple positions to None

        e.g. suppose group_attribute references the second position in the group tuple, then
        we project ('g1_value', 'g2_value', 'g3_value') to (None, 'g2_value', None)
        """
        if not group_attribute:
            return group_tuple

        assert all_group_attributes, \
            "group_by without grouping, did you specify @define_stats_grouping for event {}?".format(event)
        assert group_attribute in all_group_attributes, \
            "group {} is not configured for event {}, check @define_stats_grouping".format(group_attribute, event)

        return tuple(group_tuple[idx] if group_attribute == g else None for idx, g in enumerate(all_group_attributes))

    T = TypeVar('T')

    def create_event_topic(self, interface_class: Type[T]) -> T:
        """
        Provide an event topic proxy analogous to the event proxies provided by EventSystem/PubSub. But in contrast
        to the event system, this can be used to inject statistics also on the step, episode and epoch level.

        Note that different LogStatsLevel result in different behaviour of the returned event topic proxies!

        :param interface_class: The class object of an abstract interface that defines the events as methods.
        :return: A proxy object, dynamically derived from the passed `interface_class`, that can be used to trigger
                 events.
        """
        return event_topic_factory(interface_class, self.add_event)


class LogStatsWriter(ABC):
    """A minimal interface concrete log statistics writers must implement."""

    @abstractmethod
    def write(self, path: Optional[str], step: int, stats: LogStats) -> None:
        """
        Write the passed statistics dictionary to the log.

        :param path: This can be a path-like string to organize the log into different sections.
        :param step: The step number associated with the passed statistics
        :param stats: The statistics dictionary

        :return None
        """

    def close(self) -> None:
        """Close writer and clean up.
        """


class GlobalLogState:
    """Internal class that encapsulates the global state of the logging system."""

    global_step = 1
    global_log_stats_writers: List[LogStatsWriter] = list()

    hook_on_log_step: List[Callable] = list()
    """ list of functions called on increment_log_step() """


class LogStatsLogger(LogStatsConsumer):
    """Auxiliary class returned by get_stats_logger."""

    def receive(self, stat: LogStats) -> None:
        """Implementation of LogStatsConsumer interface
        """
        log_stats(stat, self.path)

    def __init__(self, path: Optional[str]):
        self.path = path


def register_log_stats_writer(writer: LogStatsWriter) -> None:
    """
    Set the concrete writer implementation that will receive all successive statistics logging.

    :param writer: The writer to be used by the logging system
    """
    GlobalLogState.global_log_stats_writers.append(writer)


def log_stats(stats: LogStats, path: Optional[str]) -> None:
    """Helper function.

    :param stats: The statistics dictionary
    :param path: This can be a path-like string to organize the log into different sections.
    """
    for writer in GlobalLogState.global_log_stats_writers:
        writer.write(path, GlobalLogState.global_step, stats)


def increment_log_step() -> None:
    """
    Notifies the logging system that the current step is finished.
    """
    for on_log_step in GlobalLogState.hook_on_log_step:
        on_log_step()

    GlobalLogState.global_step += 1


def get_stats_logger(path: Optional[str] = None) -> LogStatsConsumer:
    """
    Creates an object that can be used to pipe LogStatAggregator instances with the logging writers.

    Example usage:
    >>> logger = get_stats_logger("eval")
    >>> aggregator = LogStatsAggregator(LogStatsLevel.STEP, logger)
    >>> aggregator.reduce() # calculate the statistics and sent it to the registered logging writers

    :param path: The optional path to prefix the logging tags
    :return:
    """
    return LogStatsLogger(path)
