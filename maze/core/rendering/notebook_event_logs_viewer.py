"""Displaying event logs data as interactive plots in Jupyter Notebooks."""

from pathlib import Path
from typing import Union, Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

from maze.core.rendering.events_stats_renderer import EventStatsRenderer


class NotebookEventLogsViewer:
    """Event logs viewer for Jupyter Notebooks, built using ipython widgets.

    Usage: Inside a Jupyter Notebook, initialize the viewer with path to event logs directory, then call `build()`.

    This viewer offers elementary rendering functionality for event logs collected during environment rollout.
    Event logs are expected to be passed in as a set of TSV with filenames corresponding to event names, the
    default format as written out using the :class:`~maze.core.log_events.log_events_writer_tsv.LogEventsWriterTSV`.

    The logs will be passed and a set of widgets will be shown, offering options on which event to display and what
    event attribute to use as a metric.

    Statics are always aggregated on episode level -- the timeline displays mean value along with standard
    deviation (displayed as a ribbon).

    Optionally, events can be grouped by another attribute (e.g., distribution center ID in multi-echelon inventory
    environment) and aggregated on step level -- this way, we can show e.g. mean value of items stored by
    each distribution center across all the episodes. This can be configured using the widgets as well.

    :param: event_logs_dir_path: Path to directory where the event logs are stored.
    """

    def __init__(self, event_logs_dir_path: Union[str, Path]):
        self.event_logs_dir_path = event_logs_dir_path
        if isinstance(event_logs_dir_path, Path):
            self.event_logs_dir_path = event_logs_dir_path
        else:
            self.event_logs_dir_path = Path(event_logs_dir_path)

        self.event_log_options = []
        for path in list(self.event_logs_dir_path.glob("*.tsv")):
            self.event_log_options.append((path.stem, path))

        self.event_name_widget = None
        self.group_by_widget = None
        self.metric_widget = None
        self.aggregation_func_widget = None
        self.post_processing_func_widget = None

        self.renderer = EventStatsRenderer()

    def build(self) -> None:
        """Build the interactive viewer. Expected to be called in a Jupyter notebook after initialization."""
        from ipywidgets import interact, widgets

        self.event_name_widget = widgets.Dropdown(
            options=self.event_log_options,
            value=self.event_log_options[0][1],
            description='Event name:',
        )

        initial_columns = self._read_column_options(self.event_log_options[0][1])

        self.metric_widget = widgets.Dropdown(
            options=initial_columns,
            value=initial_columns[0],
            description='Metric:',
        )

        self.group_by_widget = widgets.Dropdown(
            options=self._options_with_none(initial_columns),
            description='Group by:',
        )

        self.event_name_widget.observe(self.update_column_options, 'value')

        self.aggregation_func_widget = widgets.Dropdown(
            options=self._options_with_none(EventStatsRenderer.AGGREGATION_FUNCS),
            description='Aggregate:',
        )

        self.post_processing_func_widget = widgets.Dropdown(
            options=self._options_with_none(EventStatsRenderer.POST_PROCESSING_FUNCS),
            description='Post-process:',
        )

        interact(self.render,
                 event_path=self.event_name_widget,
                 metric_name=self.metric_widget,
                 group_by=self.group_by_widget,
                 aggregation_func=self.aggregation_func_widget,
                 post_processing_func=self.post_processing_func_widget)

    def update_column_options(self, metadata: Dict[str, Any]):
        """Called by ipywidgets interact module when the user selects a different event to display. Loads
        attributes available for this event and updates the metric and group_by widget options accordingly."""
        columns = self._read_column_options(self.event_name_widget.value)

        self.group_by_widget.options = self._options_with_none(columns)
        self.group_by_widget.value = None

        self.metric_widget.options = columns
        self.metric_widget.value = columns[0]

    def render(self, event_path, **kwargs) -> None:
        """Refresh the rendered view. Called by ipywidgets on widget update."""
        df = pd.read_csv(event_path, sep="\t")
        plt.figure(figsize=(10, 6))
        self.renderer.render_timeline_stat(df, event_name=event_path.stem, **kwargs)
        plt.show()

    @staticmethod
    def _read_column_options(file_path: Path):
        """Read available columns (i.e. event attributes) from an event log file."""
        event_log = pd.read_csv(file_path, sep="\t")
        return [col for col in event_log.columns if col not in ["episode_id", "step_id", "env_time"]]

    @staticmethod
    def _options_with_none(options: List[str]):
        """Construct an options array for a dropdown ipywidget with added None as the first option."""
        none_option: Tuple[str, Optional[str]] = ("None", None)
        return [none_option] + list(zip(options, options))
