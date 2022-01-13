"""File holding tests for rollout seeding."""
import copy
from typing import Dict

import numpy as np
import pytest

from maze.core.log_stats.log_stats import LogStats, register_log_stats_writer
from maze.core.log_stats.log_stats_writer_console import LogStatsWriterConsole
from maze.test.shared_test_utils.run_maze_utils import run_maze_job


class LogStatsWriterExtract(LogStatsWriterConsole):
    """
    Log statistics writer implementation for testing peruses.
    """

    def __init__(self):
        self.data = dict()

    def write(self, path: str, step: int, stats: LogStats) -> None:
        """see LogStatsWriter.write"""

        for (event, name, groups), value in stats.items():
            tag = self._event_to_tag(event, name, groups)
            self.data[tuple(tag)] = value


def perform_rollout_seeding_test(hydra_overrides_sequential: Dict[str, str],
                                 hydra_overrides_parallel: Dict[str, str]) -> None:
    """Perform seeding test.

    :param hydra_overrides_sequential: The hydra overrides for the sequential runner.
    :param hydra_overrides_parallel: The hydra overrides for the parallel runner.
    """
    sequential_writer = LogStatsWriterExtract()
    register_log_stats_writer(sequential_writer)
    hydra_overrides_sequential.update({'runner': 'sequential', 'runner.n_episodes': 8})
    run_maze_job(hydra_overrides_sequential,
                 config_module='maze.conf', config_name='conf_rollout')
    sequential_data = copy.deepcopy(sequential_writer.data)

    parallel_writer = LogStatsWriterExtract()
    register_log_stats_writer(parallel_writer)
    hydra_overrides_parallel.update({'runner': 'parallel', 'runner.n_episodes': 8, 'runner.n_processes': 2})
    run_maze_job(hydra_overrides_parallel,
                 config_module='maze.conf', config_name='conf_rollout')
    parallel_data = copy.deepcopy(parallel_writer.data)

    for kk in sequential_data.keys():
        assert np.isclose(sequential_data[kk], parallel_data[kk]), f'Not equal stats: {kk} -> ' \
                                                                   f'{sequential_data[kk]} vs {parallel_data[kk]}'


def test_base_seeding() -> None:
    """Test rollout seeding."""
    perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '1234'},
                                 {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '1234'})

    perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '4321'},
                                 {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '4321'})

    with pytest.raises(AssertionError):
        perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '4321'},
                                     {'seeding.env_base_seed': '2345', 'seeding.agent_base_seed': '4321'})

    with pytest.raises(AssertionError):
        perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '2345'},
                                     {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '4321'})


def test_shuffle_seeds() -> None:
    """Test rollout seeding with shuffle."""
    perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '1234',
                                  'seeding.shuffle_seeds': 'true'},
                                 {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '1234',
                                  'seeding.shuffle_seeds': 'true'})

    perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '4321',
                                  'seeding.shuffle_seeds': 'true'},
                                 {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '4321',
                                  'seeding.shuffle_seeds': 'true'})

    with pytest.raises(AssertionError):
        perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '2345'},
                                     {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '2345',
                                      'seeding.shuffle_seeds': 'true'})

        perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '1234'},
                                     {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '1234',
                                      'seeding.shuffle_seeds': 'true'})


def test_explicit_seeds() -> None:
    """Test rollout seeding with explicit seeds."""
    env_seeds = list(range(50))
    agent_seeds = list(range(50))
    perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '1234',
                                  'seeding.explicit_env_seeds': env_seeds,
                                  'seeding.explicit_agent_seeds': agent_seeds},
                                 {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '1234',
                                  'seeding.explicit_env_seeds': env_seeds,
                                  'seeding.explicit_agent_seeds': agent_seeds})

    perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '4321',
                                  'seeding.explicit_env_seeds': env_seeds,
                                  'seeding.explicit_agent_seeds': agent_seeds},
                                 {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '4321',
                                  'seeding.explicit_env_seeds': env_seeds,
                                  'seeding.explicit_agent_seeds': agent_seeds})

    perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '4321',
                                  'seeding.explicit_env_seeds': env_seeds,
                                  'seeding.explicit_agent_seeds': agent_seeds},
                                 {'seeding.env_base_seed': '2345', 'seeding.agent_base_seed': '4321',
                                  'seeding.explicit_env_seeds': env_seeds,
                                  'seeding.explicit_agent_seeds': agent_seeds}, )

    perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '2345',
                                  'seeding.explicit_env_seeds': env_seeds,
                                  'seeding.explicit_agent_seeds': agent_seeds},
                                 {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '4321',
                                  'seeding.explicit_env_seeds': env_seeds,
                                  'seeding.explicit_agent_seeds': agent_seeds})


def test_explicit_seeds_shuffle_seeds() -> None:
    """Test rollout seeding with explicit seeds and shuffle."""
    env_seeds = list(range(50))
    agent_seeds = list(range(50))
    perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '1234',
                                  'seeding.shuffle_seeds': 'true',
                                  'seeding.explicit_env_seeds': env_seeds,
                                  'seeding.explicit_agent_seeds': agent_seeds},
                                 {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '1234',
                                  'seeding.shuffle_seeds': 'true',
                                  'seeding.explicit_env_seeds': env_seeds,
                                  'seeding.explicit_agent_seeds': agent_seeds})

    perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '4321',
                                  'seeding.shuffle_seeds': 'true',
                                  'seeding.explicit_env_seeds': env_seeds,
                                  'seeding.explicit_agent_seeds': agent_seeds},
                                 {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '4321',
                                  'seeding.shuffle_seeds': 'true',
                                  'seeding.explicit_env_seeds': env_seeds,
                                  'seeding.explicit_agent_seeds': agent_seeds})

    with pytest.raises(AssertionError):
        perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '2345'},
                                     {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '2345',
                                      'seeding.shuffle_seeds': 'true',
                                      'seeding.explicit_env_seeds': env_seeds,
                                      'seeding.explicit_agent_seeds': agent_seeds})

        perform_rollout_seeding_test({'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '1234'},
                                     {'seeding.env_base_seed': '1234', 'seeding.agent_base_seed': '1234',
                                      'seeding.shuffle_seeds': 'true',
                                      'seeding.explicit_env_seeds': env_seeds,
                                      'seeding.explicit_agent_seeds': agent_seeds})
