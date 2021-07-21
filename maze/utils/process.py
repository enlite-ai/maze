"""A wrapper for multiprocessing.Process that supports exception handling and return objects."""
import multiprocessing as mp
from typing import Optional, Any
from maze.utils.log_stats_utils import clear_global_state
import os
import re
import subprocess


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


def query_cpu() -> int:
    """
    Queries alloted CPUs.
    Source: https://bugs.python.org/issue36054.
    :return: Alloted CPUs.
    """

    cpu_quota = -1

    if os.path.isfile('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'):
        # Not useful for AWS Batch based jobs as result is -1, but works on local linux systems
        cpu_quota = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read().rstrip())
    if cpu_quota != -1 and os.path.isfile('/sys/fs/cgroup/cpu/cpu.cfs_period_us'):
        cpu_period = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read().rstrip())
        # Divide quota by period and you should get num of allotted CPU to the container, rounded down if fractional.
        avail_cpu = int(cpu_quota / cpu_period)
    elif os.path.isfile('/sys/fs/cgroup/cpuset/cpuset.cpus'):
        # Has potentially repeating, comma-separated groups of CPU_idx-CPU_idx or just CPU_idx.
        avail_cpu = 0
        for cpu_group in open('/sys/fs/cgroup/cpuset/cpuset.cpus').read().rstrip().split(","):
            cpu_range = cpu_group.split("-")
            avail_cpu += int(cpu_range[1] if len(cpu_range) == 2 else cpu_range[0]) - int(cpu_range[0]) + 1
    else:
        avail_cpu = os.cpu_count()

    return avail_cpu
