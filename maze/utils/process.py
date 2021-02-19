"""A wrapper for multiprocessing.Process that supports exception handling."""
import multiprocessing as mp
from typing import Optional

from maze.utils.log_stats_utils import clear_global_state


class Process(mp.Process):
    """A wrapper for multiprocessing.Process that supports exception handling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self) -> None:
        """overwrite Process.run()"""
        # clean any left-over state from the parent process
        clear_global_state()

        try:
            super().run()
            self._cconn.send(None)
        except Exception as e:
            self._cconn.send(e)
            raise e

    def exception(self) -> Optional[Exception]:
        """Get any exception that might occurred."""
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception
