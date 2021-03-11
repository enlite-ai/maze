"""Setup testing fixtures"""
import os

import matplotlib
import pytest

from maze.utils.log_stats_utils import clear_global_state


@pytest.fixture(scope="function", autouse=True)
def fixture_clear_global_state():
    """clear global state before any test"""
    clear_global_state()


@pytest.fixture(scope="function", autouse=True)
def fixture_run_in_temp_dir(tmpdir):
    """Set a new, clean, temporary working directory for every test."""
    os.chdir(tmpdir)


@pytest.fixture(scope="function", autouse=True)
def fixture_use_non_interactive_matplotlib_backend():
    """Render matplotlib plots in the non-interactive backend, which is faster and does not cause issues on a
    headless test system (gitlab runners)"""
    matplotlib.use('Agg')
