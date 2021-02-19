"""Ad-hoc statistics rendering during environment rollout loop."""

from collections import defaultdict
from typing import Optional, Callable
import numpy as np

import matplotlib.pyplot as plt

from maze.core.log_events.episode_event_log import EpisodeEventLog


class StepStatsRenderer:
    """Simple statistics rendering based on episode event logs.

    Suitable e.g. for ad-hoc plotting of statistics for the current episode during rollout."""

    @staticmethod
    def render_stats(episode_event_log: EpisodeEventLog,
                     event_name: str = "BaseEnvEvents.reward",
                     group_by: Optional[str] = None,
                     aggregation_func: Optional[Callable] = None,
                     metric: str = "value",
                     cumulative: bool = True):
        """
        Queries the event log for the given events, then aggregates them and plots them according to the
        provided options. By default, a cumulative reward is plotted.

        :param episode_event_log: The episode event log to draw events from.
        :param event_name: Name of the event to plot.
        :param group_by: Attribute of the event that the events should be grouped by when aggregating.
        :param aggregation_func: Function to aggregate the metric with.
        :param metric: The metric to plot.
        :param cumulative: If true, a cumulative sum of the metric is performed (after aggregation).
        """
        # Dictionary of events, sorted per event type
        events = defaultdict(list)
        group_by_values = set()

        # Filter events and split them per desired grouping
        for step_id, step_event_log in enumerate(episode_event_log.step_event_logs):
            for event_record in step_event_log.events:
                if event_name != event_record.interface_method.__qualname__:
                    continue
                assert group_by is None or group_by in event_record.attributes.keys(), \
                    "Group by must be present in event attributes"
                assert metric in event_record.attributes.keys(), \
                    "Metric by must be present in event attributes"

                group_by_value = event_record.attributes[group_by] if group_by is not None else None
                events[(step_id, group_by_value)].append(event_record.attributes[metric])
                group_by_values.add(group_by_value)

        # Aggregate these
        aggregated = {}
        for key, values in events.items():
            if aggregation_func is None:
                assert len(values) == 1, "Aggregation function is None but multiple values encountered"
                aggregated[key] = values[0]
            else:
                aggregated[key] = aggregation_func(values)

        # Turn into plotable arrays
        lines = {}
        step_ids = list(range(len(episode_event_log.step_event_logs)))
        for group_by_value in group_by_values:
            lines[group_by_value] = list(map(lambda step_id: aggregated.get((step_id, group_by_value), 0), step_ids))

        # Plot
        plt.figure()
        for group_by_value, series in lines.items():
            if cumulative:
                series = np.cumsum(series)
            plt.plot(step_ids, series, label=group_by_value)
        if group_by is not None:
            plt.legend(title=group_by)
        plt.title(event_name)
        plt.xlabel("Step ID")

        metric_label = metric
        if aggregation_func:
            metric_label = aggregation_func.__name__ + ": " + metric_label
        if cumulative:
            metric_label += " (cumulative)"

        plt.ylabel(metric_label)
        plt.show()
