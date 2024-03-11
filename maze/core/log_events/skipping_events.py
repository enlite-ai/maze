"""File holding events related to (sub step) skipping."""
from __future__ import annotations

from abc import ABC

import numpy as np
from maze.core.log_stats.event_decorators import define_episode_stats, define_epoch_stats, define_step_stats


class SkipEvent(ABC):
    """
    Event topic class with logging statistics to count and contextualise the skipped steps.
    """

    @define_epoch_stats(np.mean, input_name='ep_sum_flat_steps_skipped', output_name='mean_skipped')
    @define_epoch_stats(sum, input_name='ep_sum_flat_steps_skipped', output_name='sum_skipped')
    @define_episode_stats(sum, input_name='sum', output_name='ep_sum_flat_steps_skipped')
    @define_step_stats(sum, output_name='sum')
    def flat_step(self, flat_step_is_skipped: bool):
        """Event tracker to count the skipped steps at the flat level.
        :param flat_step_is_skipped: Boolean that is true if the step is skipped."""

    @define_epoch_stats(np.mean, input_name='ep_sum_sub_steps_skipped', output_name='mean_skipped')
    @define_epoch_stats(sum, input_name='ep_sum_sub_steps_skipped', output_name='sum_skipped')
    @define_episode_stats(sum, input_name='sum', output_name='ep_sum_sub_steps_skipped')
    @define_step_stats(sum, output_name='sum')
    def sub_step(self, sub_step_is_skipped: bool):
        """Event tracker to count the skipped sub steps.
        :param sub_step_is_skipped: Boolean that is true if the sub step is skipped
        """
