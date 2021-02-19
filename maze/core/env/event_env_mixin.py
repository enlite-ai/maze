"""Interface for environments to expose event logging capabilities."""
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from maze.core.events.event_record import EventRecord
from maze.core.log_events.kpi_calculator import KpiCalculator


class EventEnvMixin(ABC):
    """
    This interface provides a standard way of attaching environment events to the log statistics system.

    Implement this interface in the environment to activate the statistics support.
    """

    @abstractmethod
    def get_step_events(self) -> Iterable[EventRecord]:
        """Retrieve all recorded events of the current environment step.
        """

    @abstractmethod
    def get_kpi_calculator(self) -> Optional[KpiCalculator]:
        """If available, return an instance of a KPI calculator that can be used to calculate KPIs
        from events at the end of episode.

        :return KPI calculator or None if KPIs are not supported in this env.
        """
