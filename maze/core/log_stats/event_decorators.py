"""Event decorators can be used to compactly define logging statistics in event interfaces."""
import inspect
from collections import defaultdict
from typing import Callable, Optional, List, Dict

from maze.core.log_stats.log_stats import LogStatsLevel


def _get_all_input_names(level: LogStatsLevel, func_obj: Callable) -> List[str]:
    """
    Obtain all valid input names for a specific event method.

    :param level The level is required to include the output names of the predecessor statistics.
    :param func_obj The event method
    """
    signature = inspect.signature(func_obj)
    event_attributes = [p.name for p in list(signature.parameters.values())[1:]]

    if level is LogStatsLevel.STEP:
        return event_attributes

    if level is LogStatsLevel.EPISODE:
        previous_dict = getattr(func_obj, LogStatsLevel.STEP.name, None)
    elif level is LogStatsLevel.EPOCH:
        previous_dict = getattr(func_obj, LogStatsLevel.EPISODE.name, None)
    else:
        raise ValueError("invalid level {}".format(level))

    if not previous_dict:
        return event_attributes

    return [output_name
            for agg_list in previous_dict.values()
            for agg_fn, output_name, group_by, cumulative in agg_list
            ]


def _check_input_name(level: LogStatsLevel, func_obj: Callable, input_name: str) -> None:
    if input_name is None:
        return

    # get all available input names from the previous aggregation level
    input_names = _get_all_input_names(level, func_obj)

    # check if the input name is existing
    if input_name not in input_names:
        raise ValueError("unknown input {}, expected on of {}".format(input_name, input_names))


def _check_output_name(input_name: Optional[str],
                       output_name: Optional[str],
                       input_to_reducers: Dict[str, list]) -> str:
    # reuse the input name for the output name per default
    if output_name is None:
        output_name = input_name

    if output_name in input_to_reducers:
        if output_name is None:
            raise ValueError("non-unique output name, please specify the output_name")

        raise ValueError("duplicated output name {}".format(output_name))

    return output_name


def _decorator_factory(level: LogStatsLevel,
                       reduce_function: Callable,
                       input_name: Optional[str],
                       output_name: Optional[str],
                       group_by: Optional[str],
                       cumulative: bool = False) -> Callable:
    """
    Aggregation decorator factory

    :param level: The aggregation level that this decoration refers to
    :param reduce_function: The statistics calculation to be used
    :param input_name: Specifies the event attribute to be passed to the aggregation_function
                       (in case the event accepts more than one argument)
    :param output_name: Specify the name of the output, which is necessary when using multiple aggregations on the
                        same attribute.
    :param group_by: If there are multiple groups defined for an event, per default the statistics
                     is collected at the 'cell' level. This option allows to project the statistics
                     onto a single group (e.g. for inventory statistics we might define location and product as
                     groups, but prefer to monitor statistics grouped only per product, regardless of location
                     and vice versa)
    :param cumulative: Enable cumulative statistics

    :return: The decorator function
    """

    def decorator(func_obj) -> Callable:
        """

        @param func_obj: The actual function to be decorated
        @return: The unmodified function object (required by the Python decorator semantics)
        """
        # attach a dictionary that maps the aggregation level to the statistics calculation info
        input_to_reducers = getattr(func_obj, level.name, None)

        # create the aggregation dictionary if it does not exist (this will be the case for the first decorator)
        if not input_to_reducers:
            input_to_reducers = defaultdict(list)
            setattr(func_obj, level.name, input_to_reducers)

        # check if the input_name is valid
        _check_input_name(level, func_obj, input_name)

        # check if the output_name is valid
        _output_name = _check_output_name(input_name=input_name,
                                          output_name=output_name,
                                          input_to_reducers=input_to_reducers)

        # add the new aggregation to the dictionary
        input_to_reducers[input_name].append((reduce_function, _output_name, group_by, cumulative))

        return func_obj

    return decorator


def define_step_stats(reduce_function: Optional[Callable],
                      input_name: Optional[str] = None,
                      output_name: Optional[str] = None,
                      group_by: Optional[str] = None,
                      cumulative: bool = False) -> Callable:
    """
    Event method decorator, defines a new step statistics calculation for this event.

    :input: all events in a single step (and side-loaded step statistics, see 'reduce_function' set to None)
    :output: step statistics

    :param reduce_function: A function that takes a list of values and returns the calculated statistics.
                            In the special case that we do not want to calculate the statistics from events,
                            but have the statistics result already available for the current step, the reduce_function
                            can be set to None. Then the event can be invoked at most once per step to side-load the
                            result. This is very useful to log state information, e.g. the inventory size.
    :param input_name: The name of the event attribute (=keyword attribute), whose values are to be passed
                       to the reduce function.
                       Can be omitted, for which there are two reasons

                       * no naming necessary, there is only a single event attribute
                       * we want all event attributes to be passed to the reduce_function as dictionaries (or our
                         reduce function does not care, e.g. counting the number of events with `len`)

    :param output_name: The name of the statistics, how it should be passed to the logger resp. the following
                        aggregation stage. Will be the same as input_name if None is provided.
    :param group_by: If there are multiple groups defined for an event, per default the statistics
                     is collected at the 'cell' level. This option allows to project the statistics
                     onto a single group (e.g. for inventory statistics we might define location and product as
                     groups, but prefer to monitor statistics grouped only per product, regardless of location
                     and vice versa)
    :param cumulative: Enable cumulative statistics

    :return: The decorator function
    """
    return _decorator_factory(LogStatsLevel.STEP, reduce_function, input_name, output_name, group_by, cumulative)


def define_episode_stats(reduce_function: Callable,
                         input_name: Optional[str] = None,
                         output_name: Optional[str] = None,
                         group_by: Optional[str] = None,
                         cumulative: bool = False) -> Callable:
    """
    Event method decorator, defines a new episode statistics calculation for this event.

    :input: all step statistics in the current episode
            and side-loaded episode statistics, see 'reduce_function' set to None
    :output: episode statistics


    :param reduce_function: A function that takes a list of values and returns the calculated statistics
    :param input_name: The name of the step statistics, whose values are to be passed to the reduce function.
                       Can be omitted, for which there are two reasons

                       * no naming necessary, there is only a single step statistics available
                       * we want all step statistics to be passed to the reduce_function as dictionaries (or our
                         reduce function does not care, e.g. counting with `len`)

    :param output_name: The name of the generated episode statistics, how it should be passed to the logger respective
                        the following aggregation stage. Will be the same as input_name if None is provided.
    :param group_by: If there are multiple groups defined for an event, per default the statistics
                     is collected at the 'cell' level. This option allows to project the statistics
                     onto a single group (e.g. for inventory statistics we might define location and product as
                     groups, but prefer to monitor statistics grouped only per product, regardless of location
                     and vice versa)
    :param cumulative: Enable cumulative statistics

    :return: The decorator function
    """
    return _decorator_factory(LogStatsLevel.EPISODE, reduce_function, input_name, output_name, group_by, cumulative)


def define_epoch_stats(reduce_function: Callable,
                       input_name: Optional[str] = None,
                       output_name: Optional[str] = None,
                       group_by: Optional[str] = None,
                       cumulative: bool = False) -> Callable:
    """
    Event method decorator, defines a new epoch statistics calculation for this event.

    :input: All episode statistics of the current epoch and side-loaded epoch statistics,
            see 'reduce_function' set to None

    :output: Epoch statistics

    :param reduce_function: A function that takes a list of values and returns the calculated statistics
    :param input_name: The name of the event attribute (=keyword attribute), whose values are to be passed
                       to the reduce function.
                       Can be omitted, for which there are two reasons

                       * no naming necessary, there is only a single epoch statistics available
                       * we want all episode statistics to be passed to the reduce_function as dictionaries (or our
                         reduce function does not care, e.g. counting with `len`)

    :param output_name: The name of the generated epoch statistics, how it should be passed to the logger respective
                        the following aggregation stage. Will be the same as input_name if None is provided.
    :param group_by: If there are multiple groups defined for an event, per default the statistics
                     is collected at the 'cell' level. This option allows to project the statistics
                     onto a single group (e.g. for inventory statistics we might define location and product as
                     groups, but prefer to monitor statistics grouped only per product, regardless of location
                     and vice versa)
    :param cumulative: Enable cumulative statistics

    :return: The decorator function
    """
    return _decorator_factory(LogStatsLevel.EPOCH, reduce_function, input_name, output_name, group_by, cumulative)


def define_stats_grouping(*group_by: str) -> Callable:
    """
    Event method decorator, defines a grouping of all calculated statistics by an attribute.

    :param group_by: Name of the event attribute(s)

    :return: The decorator function
    """

    def decorator(func_obj):
        """
        :param func_obj: The actual function to be decorated
        :return The unmodified function object (required by the Python decorator semantics)
        """
        existing_group_by = getattr(func_obj, "group_by", None)
        assert existing_group_by is None, "more than one stats_grouping decorator detected"

        setattr(func_obj, "group_by", group_by)

        return func_obj

    return decorator


def define_plot(create_figure_function: Callable, input_name: str = None) -> Callable:
    """
    Event method decorator, defines a plot.
    :return: The decorator function
    """

    def decorator(func_obj):
        """
        :param func_obj: The actual function to be decorated
        :return The unmodified function object (required by the Python decorator semantics)
        """
        render_figure_dict = getattr(func_obj, "tensorboard_render_figure_dict", None)

        # create dict if not existing
        if not render_figure_dict:
            render_figure_dict = dict()
            setattr(func_obj, "tensorboard_render_figure_dict", render_figure_dict)

        render_figure_dict[input_name] = create_figure_function

        return func_obj

    return decorator
