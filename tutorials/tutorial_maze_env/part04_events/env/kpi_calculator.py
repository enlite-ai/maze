from typing import Dict

from maze.core.env.maze_state import MazeStateType
from maze.core.log_events.kpi_calculator import KpiCalculator
from maze.core.log_events.episode_event_log import EpisodeEventLog
from .events import InventoryEvents


class Cutting2dKpiCalculator(KpiCalculator):
    """KPIs for 2D cutting environment.
    The following KPIs are available: Raw pieces used per step
    """

    def calculate_kpis(self, episode_event_log: EpisodeEventLog, last_maze_state: MazeStateType) -> Dict[str, float]:
        """Calculates the KPIs at the end of episode."""

        # get overall step count of episode
        step_count = len(episode_event_log.step_event_logs)
        # count raw inventory piece replenishment events
        raw_piece_usage = 0
        for _ in episode_event_log.query_events(InventoryEvents.piece_replenished):
            raw_piece_usage += 1
        # compute step normalized raw piece usage
        return {"raw_piece_usage_per_step": raw_piece_usage / step_count}
