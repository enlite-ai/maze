from abc import ABC

import numpy as np

from maze.core.log_stats.event_decorators import define_step_stats, define_episode_stats, define_epoch_stats, \
    define_stats_grouping


class EnvProfilingEvents(ABC):
    """
    Event topic class with logging statistics based only on observations, therefore applicable to any valid
    reinforcement learning environment.
    """
    @define_epoch_stats(np.mean, input_name="st_ep_mean", output_name="sub_mean")
    @define_epoch_stats(np.mean, input_name="fs_ep_count", output_name="flat_mean")
    @define_epoch_stats(sum, input_name="ep_count", output_name="sub_count")
    @define_episode_stats(np.mean, input_name='st_mean', output_name="st_ep_mean")
    @define_episode_stats(np.mean, input_name='fs_mean', output_name="fs_ep_count")
    @define_episode_stats(sum, input_name='len', output_name="ep_count")
    @define_step_stats(np.mean, output_name='st_mean')
    @define_step_stats(sum, output_name='fs_mean')
    @define_step_stats(len, output_name='len')
    def full_env_step_time(self, value: float):
        """Record the full env step time in seconds."""

    @define_epoch_stats(np.mean, input_name="ep_mean", output_name="mean")
    @define_episode_stats(np.mean, input_name='st_mean', output_name="ep_mean")
    @define_step_stats(np.mean, input_name='per', output_name='st_mean')
    @define_stats_grouping("wrapper_name")
    def wrapper_step_time(self, wrapper_name: str, time: float, per: float):
        """Record the wrapper step time in percent wrt to the full env step."""

    @define_epoch_stats(np.mean, input_name="ep_mean", output_name="mean")
    @define_episode_stats(np.mean, output_name="ep_mean")
    @define_step_stats(np.mean, input_name='per')
    def maze_env_step_time(self, time: float, per: float):
        """Record the maze env step time in percent wrt to the full env step."""

    @define_epoch_stats(np.mean, input_name="ep_mean", output_name="mean")
    @define_episode_stats(np.mean, output_name="ep_mean")
    @define_step_stats(np.mean, input_name='per')
    def core_env_step_time(self, time: float, per: float):
        """Record the core env step time in percent wrt to the full env step."""

    @define_epoch_stats(np.mean, input_name="ep_mean", output_name="mean")
    @define_episode_stats(np.mean, output_name="ep_mean")
    @define_step_stats(np.mean, input_name='per')
    def observation_conv_time(self, time: float, per: float):
        """Record the observation conversion time in percent wrt to the full env step."""

    @define_epoch_stats(np.mean, input_name="ep_mean", output_name="mean")
    @define_episode_stats(np.mean, output_name="ep_mean")
    @define_step_stats(np.mean, input_name='per')
    def action_conv_time(self, time: float, per: float):
        """Record the action conversion time in percent wrt to the full env step."""

    @define_epoch_stats(np.mean, input_name="ep_mean", output_name="per_of_core_env")
    @define_episode_stats(np.mean, input_name='time', output_name="ep_mean")
    @define_step_stats(np.mean, input_name='per', output_name='time')
    @define_stats_grouping("name")
    def investigate_time(self, name: str, time: float, per: float):
        """In case anywhere in the wrappers, maze_env or core env the attribute self._investigate_relative_time is
        written it will be divided by the total step time and logged with this event.."""

