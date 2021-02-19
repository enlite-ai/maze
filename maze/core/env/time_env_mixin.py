"""Interface for environments to expose internal environment time."""

from abc import ABC, abstractmethod


class TimeEnvMixin(ABC):
    """This interface provides a standard way of exposing environment time to external components and wrappers.
    e.g. for event logging.
    """

    @abstractmethod
    def get_env_time(self) -> int:
        """
        :return: Internal environment time represented as integer.
        """
