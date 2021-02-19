"""Setup testing fixtures"""
import os

import pytest
from maze.utils.log_stats_utils import clear_global_state


@pytest.fixture(scope="function", autouse=True)
def maze_core_clear_global_state():
    """clear global state before any test"""
    clear_global_state()


@pytest.fixture(scope="function", autouse=True)
def maze_core_run_in_temp_dir(tmpdir):
    """Set a new, clean, temporary working directory for every test."""
    os.chdir(tmpdir)
