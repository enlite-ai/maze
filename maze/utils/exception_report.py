"""Helper class for reporting exceptions between processes."""

import traceback


class ExceptionReport:
    """Data class for reporting exceptions between processes."""

    def __init__(self, exception: Exception):
        self.exception = exception
        self.traceback = traceback.format_exc()
