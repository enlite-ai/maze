"""Implementation of Tensorboard output for the logging statistics system."""
import logging
import math
from typing import Callable, Union, List, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from maze.core.annotations import override
from maze.core.log_stats.log_stats import LogStatsWriter, LogStats, GlobalLogState


class LogStatsWriterTensorboard(LogStatsWriter):
    """
    Log statistics writer implementation for Tensorboard.
    :param log_dir: log_dir for TensorBoard
    :param tensorboard_render_figure: Indicates whether to visualize the actions taken in TensorBoard.
    """

    def __init__(self, log_dir: str, tensorboard_render_figure: bool):
        self.tensorboard_render_figure = tensorboard_render_figure

        self.summary_writer = SummaryWriter(log_dir=log_dir)

        self.previous_step_tags = set()
        self.this_step_tags = set()
        GlobalLogState.hook_on_log_step.append(self.on_log_step_increment)

        class _IgnoreTensorboardCheckNaN(logging.Filter):
            def filter(self, record):
                """Filter out a logging warning message caused by a unnecessary NaN check in tensorboard/x2num.py"""
                return not (record.funcName == "check_nan" and record.msg == "NaN or Inf found in input tensor.")

        # this relies on knowing the variable name in my_module
        logging.getLogger().addFilter(_IgnoreTensorboardCheckNaN())

    @override(LogStatsWriter)
    def write(self, path: str, step: int, stats: LogStats) -> None:
        """LogStatsWriter.write implementation"""

        # Create custom figures for TensorBoard visualization
        # ===================================================
        if self.tensorboard_render_figure:
            for (event, name, groups), value in stats.items():
                tag = self._event_to_tag(event, name, groups)
                if path:
                    tag = path.replace("/", "_") + "_" + tag

                # get plotting function from the event definition
                render_figure_dict = getattr(event, "tensorboard_render_figure_dict", dict())
                render_figure_function = render_figure_dict.get(name, None)
                if render_figure_function:
                    fig = render_figure_function(value, event=event, name=event, groups=groups)
                    self.summary_writer.add_figure(tag=tag, figure=fig, global_step=step)

            self.summary_writer.flush()

        # Vanilla TensorBoard visualization
        # =================================
        for (event, name, groups), value in stats.items():
            tag = self._event_to_tag(event, name, groups)
            if path:
                tag = path.replace("/", "_") + "_" + tag

            # Skip all events that have the attribute "tensorboard_render_figure". Events with this attribute are
            # plotted with an external plotting library and added to TensorBoard manually above.
            if getattr(event, "tensorboard_render_figure_dict", dict()):
                continue

            if isinstance(value, List):
                self.summary_writer.add_histogram(tag, np.array(value), step)
            elif isinstance(value, float) or isinstance(value, int):
                self.summary_writer.add_scalar(tag, value, step)
            else:
                # Only lists and scalars are added to TensorBoard
                pass
            self.this_step_tags.add(tag)

        self.summary_writer.flush()

    def on_log_step_increment(self):
        """Hooked into increment_log_step, called by the logging system immediately before the increment."""
        # write nan values explicitly to stop tensorboard from interpolating missing values in the graphs
        for tag in self.previous_step_tags:
            if tag not in self.this_step_tags:
                self.summary_writer.add_scalar(tag, math.nan, GlobalLogState.global_step)

        # update previous tags and reset tags for next step
        self.previous_step_tags = self.this_step_tags
        self.this_step_tags = set()

        self.summary_writer.flush()

    @override(LogStatsWriter)
    def close(self) -> None:
        """LogStatsWriter.write implementation"""
        self.summary_writer.close()

    @staticmethod
    def _event_to_tag(event: Callable, name: str, groups: Optional[List[Union[int, str]]]) -> str:
        # use the qualified name of the method as a basis, in the form 'EventInterface.event_method'
        qualified_name = event.__qualname__

        key_name = qualified_name.replace('.', '/')

        if name is not None and len(name):
            key_name = key_name + "/" + name

        if groups is not None:
            for group in groups:
                # support grouping projections by skipping None values
                if group is None:
                    continue

                key_name = key_name + "/" + str(group)

        return key_name
