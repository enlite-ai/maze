"""Rendering customized statistics on top of event logs."""

import re
from typing import Optional, Callable, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from maze.core.log_events.episode_event_log import EpisodeEventLog


class EventStatsRenderer:
    """Renders customizable statistics on top of event logs.

    This renderer provides a central rendering functionality for event log data. Elementary customizability
    is offered (e.g. simple aggregation etc.). For more complex operations with the data, it is advised
    to work with the TSV event logs directly.
    """

    AGGREGATION_FUNCS = ["mean", "sum", "min", "max", "count"]
    """Aggregation functions to offer to the user. Recognized as strings by pandas."""
    POST_PROCESSING_FUNCS = ["cumsum"]
    """Post-processing functions to offer to the user. Recognized as strings by pandas."""

    def __init__(self):
        sns.set_style("darkgrid")
        self.figure = None

    @staticmethod
    def render_timeline_stat(df: pd.DataFrame,
                             event_name: str = "BaseEnvEvents.reward",
                             metric_name: str = "value",
                             aggregation_func: Optional[Union[str, Callable]] = None,
                             group_by: str = None,
                             post_processing_func: Optional[Union[str, Callable]] = 'cumsum'):
        """Render event statistics from a data frame according to the supplied options.

        Does not create a figure, renders into the currently active ax.

        :param df: Event log to render statistics from
        :param event_name: Name of the even the even log corresponds to
        :param metric_name: Metric to use (one of the event attributes, e.g. "n_items" -- depends on the event type)
        :param aggregation_func: Optionally, specifies how to aggregate the metric on step level, i.e. when there
                                 are multiple same events dispatched during the same step.
        :param group_by: Optionally, another of event attributes to group by on the step level (e.g. "product_id")
        :param post_processing_func: Optionally, a function to post-process the data ("cumsum" is often used)
        """
        assert metric_name != group_by, "group by attribute cannot be the same as metric attribute"
        assert metric_name in df.columns, "metric by must be present in event attributes"
        assert group_by is None or group_by in df.columns, "group by must be none or present in event attributes"

        grouping = ["episode_id", "env_time"]
        if group_by in df.columns:
            grouping.append(group_by)

        # Aggregate
        if aggregation_func is not None:
            df = df.groupby(grouping, as_index=False).agg({metric_name: aggregation_func})

        # Post-process
        if post_processing_func is not None:
            df[metric_name] = df.groupby(grouping, as_index=False)[metric_name].transform(post_processing_func)

        sns.lineplot(x="env_time", y=metric_name, hue=group_by, data=df)

        # Add plot labels
        event_name_components = list(map(lambda x: EventStatsRenderer._humanize(x), event_name.split(".")))
        plt.title(": ".join(event_name_components))
        ylabel_suffix = f' ({aggregation_func})' if aggregation_func is not None else ""
        plt.xlabel(EventStatsRenderer._humanize("env_time"))
        plt.ylabel(EventStatsRenderer._humanize(metric_name) + ylabel_suffix)
        plt.xlim([0, None])

    def render_current_episode_stats(self,
                                     episode_event_log: EpisodeEventLog,
                                     event_name: str = "BaseEnvEvents.reward",
                                     metric_name: str = "value",
                                     aggregation_func: Optional[Union[str, Callable]] = None,
                                     group_by: str = None,
                                     post_processing_func: Optional[Union[str, Callable]] = 'cumsum'):
        """
        Render event stats from episode log of currently running episode.

        Creates a new figure if needed.

        :param episode_event_log: Episode event log to render events from
        :param event_name: Name of the even the even log corresponds to
        :param metric_name: Metric to use (one of the event attributes, e.g. "n_items" -- depends on the event type)
        :param aggregation_func: Optionally, specifies how to aggregate the metric on step level, i.e. when there
                                 are multiple same events dispatched during the same step.
        :param group_by: Optionally, another of event attributes to group by on the step level (e.g. "product_id")
        :param post_processing_func: Optionally, a function to post-process the data ("cumsum" is often used)
        """

        assert len(episode_event_log.step_event_logs) > 0, "rendered episode log cannot be empty"

        events = []
        for step_event_log in episode_event_log.step_event_logs:
            for event_record in step_event_log.events:
                if event_name != event_record.interface_method.__qualname__:
                    continue
                assert metric_name in event_record.attributes.keys(), \
                    "metric by must be present in event attributes"
                assert group_by is None or group_by in event_record.attributes.keys(), \
                    "group by must be none or present in event attributes"

                events.append([episode_event_log.episode_id,
                               step_event_log.env_time,
                               event_record.attributes[metric_name]])

        df = pd.DataFrame(events, columns=["episode_id", "env_time", metric_name])

        if self.figure is None:
            self.figure = plt.figure()
        else:
            plt.figure(self.figure.number)
            plt.clf()
        self.render_timeline_stat(df, event_name, metric_name, aggregation_func, group_by, post_processing_func)
        plt.draw()
        plt.pause(0.1)

    def close(self):
        """Close the stats figure if one has been created."""
        if self.figure is not None:
            plt.figure(self.figure.number)
            plt.close()

    @staticmethod
    def _humanize(string: str):
        """Converts camel-case and snake-case strings into capitalised space-separated ones."""
        # Snake case -> spaces
        string = string.replace("_", " ")
        # CamelCase -> spaces
        string = re.sub(r'(?<!^)(?=[A-Z])', ' ', string)
        # Capital first letter
        string = string.lower().capitalize()
        # Special case: Detect ID, as it is rather common (in step ID)
        string = re.sub(r'(?<= )id(?=$| )', 'ID', string)
        return string
