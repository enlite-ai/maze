"""Timeout class to avoid blocking code, especially useful for testing."""
import signal


class Timeout:
    """Timeout class, fires a TimeoutError after the given number of seconds elapsed.

    Example usage:

    with Timeout(seconds=12):
        potentially_slow_function()
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def _handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        signal.alarm(0)
