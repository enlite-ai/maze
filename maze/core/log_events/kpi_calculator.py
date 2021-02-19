"""KPIs can be calculated at the end of the episode to help with measuring agent performance."""

from abc import ABC, abstractmethod
from typing import Dict

from maze.core.env.maze_state import MazeStateType
from maze.core.log_events.episode_event_log import EpisodeEventLog


class KpiCalculator(ABC):
    """Interface for calculating KPI metrics.
    If available, is called by statistics wrapper at the end of each episode."""

    @abstractmethod
    def calculate_kpis(self, episode_event_log: EpisodeEventLog, last_maze_state: MazeStateType) -> Dict[str, float]:
        """Compute KPIs for the current episode.

        This is expected to be called once at the end of the episode, if statistics logging is enabled.

        :param episode_event_log: Log of events recorded during the past episode.
        :param last_maze_state: State of the environment at the end of the episode
        :return: Values of KPI metrics in the format {kpi_name: kpi_value}
        """
