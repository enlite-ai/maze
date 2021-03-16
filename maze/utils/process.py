"""A wrapper for multiprocessing.Process that supports exception handling and return objects."""
import multiprocessing as mp
from typing import Optional, Any

from maze.utils.log_stats_utils import clear_global_state


class Process(mp.Process):
    """A wrapper for multiprocessing.Process that supports exception handling and return objects."""

    def __init__(self, target, args=(), kwargs=None):
        super().__init__(target=target, args=args, kwargs=kwargs or {})
        if kwargs is None:
            kwargs = dict()
        if kwargs is None:
            kwargs = {}
        self._target = target
        self._args = args
        self._kwargs = kwargs

        self._pconn, self._cconn = mp.Pipe()
        self._exception = None
        self._result = None

    def run(self) -> None:
        """overwrite Process.run()"""
        # clean any left-over state from the parent process
        clear_global_state()

        try:
            result = self._target(*self._args, **self._kwargs)
            self._cconn.send(result)
        except Exception as e:
            self._cconn.send(e)
            raise e

    def _poll(self):
        if self._pconn.poll():
            result = self._pconn.recv()
            if isinstance(result, Exception):
                self._exception = result
            else:
                self._result = result

    def result(self) -> Any:
        """Returns the result (if any)"""
        return self._result

    def exception(self) -> Optional[Exception]:
        """Get any exception that might occurred."""
        self._poll()

        return self._exception
