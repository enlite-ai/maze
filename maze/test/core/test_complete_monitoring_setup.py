"""Tests integration of trajectory data and event logging data together."""

from maze.core.log_events.episode_event_log import EpisodeEventLog
from maze.core.log_events.log_events_writer import LogEventsWriter
from maze.core.log_events.log_events_writer_registry import LogEventsWriterRegistry
from maze.core.trajectory_recording.records.trajectory_record import StateTrajectoryRecord
from maze.core.trajectory_recording.writers.trajectory_writer import TrajectoryWriter
from maze.core.trajectory_recording.writers.trajectory_writer_registry import TrajectoryWriterRegistry
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.trajectory_recording_wrapper import TrajectoryRecordingWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env


def test_matches_episode_ids_with_event_logs():
    class TestTrajectoryWriter(TrajectoryWriter):
        """Mock trajectory data writer. Logs episode IDs."""

        def __init__(self):
            self.episode_ids = []
            self.step_counts = []

        def write(self, episode_record: StateTrajectoryRecord):
            self.episode_ids.append(episode_record.id)
            self.step_counts.append(len(episode_record.step_records) - 1)  # Subtract the final state!

    class TestLogEventsWriter(LogEventsWriter):
        """Mock event data writer. Logs episode IDs."""

        def __init__(self):
            self.episode_ids = []
            self.step_counts = []

        def write(self, episode_record: EpisodeEventLog):
            self.episode_ids.append(episode_record.episode_id)
            self.step_counts.append(len(episode_record.step_event_logs))

    # Register mock trajectory writer
    trajectory_writer = TestTrajectoryWriter()
    TrajectoryWriterRegistry.writers = []  # Ensure there is no other writer
    TrajectoryWriterRegistry.register_writer(trajectory_writer)

    # Register mock event log writer
    event_log_writer = TestLogEventsWriter()
    LogEventsWriterRegistry.writers = []  # Ensure there is no other writer
    LogEventsWriterRegistry.register_writer(event_log_writer)

    # Test run
    env = build_dummy_maze_env()
    env = TrajectoryRecordingWrapper.wrap(env)
    env = LogStatsWrapper.wrap(env)
    for _ in range(5):
        env.reset()
        for _ in range(10):
            env.step(env.action_space.sample())

    # final env reset required
    env.reset()

    assert len(trajectory_writer.episode_ids) == 5
    assert trajectory_writer.episode_ids == event_log_writer.episode_ids
    assert trajectory_writer.step_counts == event_log_writer.step_counts
