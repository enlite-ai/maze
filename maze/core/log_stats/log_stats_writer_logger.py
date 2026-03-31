"""Implementation of logger based output for the logging statistics system."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from maze.core.log_stats.log_stats import LogStats, LogStatsWriter

import numpy as np

logger = logging.getLogger('EVENTS')
logger.setLevel(logging.INFO)


class LogStatsWriterLogger(LogStatsWriter):
    """
    Log statistics writer implementation for the logger, mainly for debugging purposes. Creates table-like logfile
    text in a fixed width layout.

    Also dumps a csv file with the aggregated statistics for each epoch.
    """

    def write(self, path: str, step: int, stats: LogStats) -> None:
        """see LogStatsWriter.write"""

        # print run directory
        exp_dir = os.path.abspath('.')
        logger.info(f'Output directory: {exp_dir}')

        # print stats
        logger.info('{:>5}|{:<104}|{:>20}'.format('step', 'path', 'value'))
        logger.info('{:>5}|{:<104}|{:>20}'.format('=' * 5, '=' * 104, '=' * 20))

        lines = []
        for (event, name, groups), value in stats.items():
            tag = self._event_to_tag(event, name, groups)

            if path:
                tag.insert(0, path)

            # limit the "tag path" to 4 elements (e.g. train|valid, EventClass, event_method, value/group)
            if len(tag) > 4:
                tag = tag[:3] + ['/'.join(tag[3:])]

            tag_full_length = tag
            tag = [self._limit(p, 22 if idx < 2 else 30) for idx, p in enumerate(tag)]
            for _ in range(4 - len(tag)):
                tag.append('\u00b7' * 22)

            if isinstance(value, list):
                # if tuple is provided extract actual value as the entry at position 0
                if isinstance(value[0], tuple):
                    value = [v[0] for v in value]
                if isinstance(value[0], str):
                    value_str = f'[len:{len(value)}, unique:{len(np.unique(value)):.1f}]'
                else:
                    value_str = f'[len:{len(value)}, μ:{np.mean(value):.1f}]'

                logger.info('{:>5}|{:<104}|{:>20}'.format(step, ''.join(tag), value_str))
                lines.append('\n{},{},{}'.format(step, ','.join(tag_full_length), value_str))
            else:
                logger.info('{:>5}|{:<104}|{:>20.3f}'.format(step, ''.join(tag), value))
                lines.append('\n{},{},{}'.format(step, ','.join(tag_full_length), value))

        with open(f'aggregated_event_logs_{step}.csv', 'w') as f:
            f.writelines(lines)

    @classmethod
    def _limit(cls, s, max_len):
        if len(s) < max_len:
            return s + ' ' * (max_len - len(s))

        return s[: max_len - 2] + '..'

    @staticmethod
    def _event_to_tag(event: Callable, name: str, groups: list[str] | None) -> list[str]:
        # use the qualified name of the method as a basis, in the form 'EventInterface.event_method'
        qualified_name = event.__qualname__

        path = qualified_name.split('.')

        if name is not None and len(name):
            path.append(name)

        if groups is not None:
            for group in groups:
                # support grouping projections by skipping None values
                if group is None:
                    continue

                path.append(str(group))

        return path
