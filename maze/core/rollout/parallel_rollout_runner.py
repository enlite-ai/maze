"""Parallel rollout runner for running envs and agents in multiple processes."""

import traceback
from collections import namedtuple
from multiprocessing import Queue, Process
from typing import Iterable, Tuple

from omegaconf import DictConfig
from tqdm import tqdm

from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.log_events.episode_event_log import EpisodeEventLog
from maze.core.log_events.log_events_writer import LogEventsWriter
from maze.core.log_events.log_events_writer_registry import LogEventsWriterRegistry
from maze.core.log_events.log_events_writer_tsv import LogEventsWriterTSV
from maze.core.log_stats.log_stats import LogStatsAggregator, LogStatsLevel, get_stats_logger
from maze.core.log_stats.log_stats import LogStatsConsumer, LogStats
from maze.core.log_stats.log_stats import register_log_stats_writer
from maze.core.log_stats.log_stats_writer_console import LogStatsWriterConsole
from maze.core.rollout.rollout_runner import RolloutRunner
from maze.core.trajectory_recording.writers.trajectory_writer_file import TrajectoryWriterFile
from maze.core.trajectory_recording.writers.trajectory_writer_registry import TrajectoryWriterRegistry
from maze.core.utils.factory import ConfigType, CollectionOfConfigType
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.trajectory_recording_wrapper import TrajectoryRecordingWrapper

EpisodeStatsReport = namedtuple("EpisodeStatsReport", "stats event_log")
"""Tuple for passing episode stats from workers to the main process."""

ExceptionReport = namedtuple("ExceptionReport", "exception traceback")
"""Tuple for passing error reports from the workers to the main process."""


class EpisodeRecorder(LogStatsConsumer, LogEventsWriter):
    """Keeps the statistics and event logs from the last episode so that it can then be shipped to the main process."""

    def __init__(self):
        self.last_stats = None
        self.last_event_log = None

    @override(LogStatsConsumer)
    def receive(self, stat: LogStats) -> None:
        """Receive the statistics from the env and store them."""
        self.last_stats = stat

    @override(LogEventsWriter)
    def write(self, episode_event_log: EpisodeEventLog) -> None:
        """Receive the event logs from the env and store them."""
        self.last_event_log = episode_event_log

    def get_last_episode_data(self) -> Tuple[LogStats, EpisodeEventLog]:
        """Get the stats and event log from the last episode.

        :return: Tuple of (episode stats, event log)
        """
        return self.last_stats, self.last_event_log


class ParallelRolloutWorker:
    """Class encapsulating functionality performed in worker processes."""

    @staticmethod
    def run(env_config: DictConfig,
            wrapper_config: DictConfig,
            agent_config: DictConfig,
            n_episodes: int,
            max_episode_steps: int,
            record_trajectory: bool,
            input_directory: str,
            reporting_queue: Queue,
            env_instance_seed: int,
            agent_instance_seed: int) -> None:
        """Build the environment and run the rollout for the specified number of episodes.

        :param env_config: Hydra configuration of the environment to instantiate.
        :param wrapper_config: Hydra configuration of environment wrappers.
        :param agent_config: Hydra configuration of agent's policies.
        :param n_episodes: Number of episodes to run (in total, will be split across processes).
        :param max_episode_steps: Max number of steps per episode to perform
                                    (episode might end earlier if env returns done).
        :param record_trajectory: Whether to record trajectory data.
        :param input_directory: Directory to load the model from.
        :param reporting_queue: Queue for passing the stats and event logs back to the main process after each episode.
        :param env_instance_seed: The seed for this particular env.
        :param agent_instance_seed: The seed for this particular agent.
        """
        try:
            env, agent = RolloutRunner.init_env_and_agent(env_config, wrapper_config, max_episode_steps,
                                                          agent_config, input_directory, env_instance_seed,
                                                          agent_instance_seed)
            env, episode_recorder = ParallelRolloutWorker._setup_monitoring(env, record_trajectory)

            RolloutRunner.run_interaction_loop(
                env, agent, n_episodes,
                episode_end_callback=lambda: reporting_queue.put(episode_recorder.get_last_episode_data())
            )
        except Exception as exception:
            # Ship exception along with a traceback to the main process
            exception_report = ExceptionReport(exception, traceback.format_exc())
            reporting_queue.put(exception_report)
            raise

    @staticmethod
    def _setup_monitoring(env: StructuredEnv, record_trajectory: bool) -> Tuple[StructuredEnv, EpisodeRecorder]:
        """Set up monitoring wrappers.

        Stats and event logs are collected in the episode recorder, so that they can be shipped to the main
        process on end of each episode.
        """
        if not isinstance(env, LogStatsWrapper):
            env = LogStatsWrapper.wrap(env)
        episode_recorder = EpisodeRecorder()
        LogEventsWriterRegistry.register_writer(episode_recorder)
        env.episode_stats.register_consumer(episode_recorder)

        # Trajectory recording happens in the worker process
        # Hydra handles working directory
        if record_trajectory:
            TrajectoryWriterRegistry.register_writer(TrajectoryWriterFile(log_dir="./trajectory_data"))
            if not isinstance(env, TrajectoryRecordingWrapper):
                env = TrajectoryRecordingWrapper.wrap(env)

        return env, episode_recorder


class ParallelRolloutRunner(RolloutRunner):
    """Runs rollout in multiple processes in parallel.

    Both agent and environment are run in multiple instances across multiple processes. While this greatly speeds
    up the rollout, the memory consumption might be high for large environments and agents.

    Trajectory recording, event logging, as well as stats logging are supported. Trajectory logging happens
    in the child processes. Event logs and stats are shipped back to the main process so that they can be
    handled together there. This allows monitoring of progress and calculation of summary stats across
    all the processes.

    (Note that the relevant wrappers need to be present in the config for the trajectory/event/stats logging to work.
    Data are logged into the working directory managed by hydra.)

    In case of early rollout termination using a keyboard interrupt, data for all episodes completed till that
    point will be preserved (= written out). Graceful shutdown will be attempted, including calculation of statistics
    across the episodes completed before the rollout was terminated.

    :param n_episodes: Count of episodes to run
    :param max_episode_steps: Count of steps to run in each episode (if environment returns done, the episode
                                will be finished earlier though)
    :param n_processes: Count of processes to spread the rollout across.
    :param record_trajectory: Whether to record trajectory data
    :param record_event_logs: Whether to record event logs
    """

    def __init__(self,
                 n_episodes: int,
                 max_episode_steps: int,
                 n_processes: int,
                 record_trajectory: bool,
                 record_event_logs: bool):
        super().__init__(n_episodes, max_episode_steps, record_trajectory, record_event_logs)
        self.n_processes = n_processes
        self.epoch_stats_aggregator = None
        self.reporting_queue = None

    @override(RolloutRunner)
    def run_with(self, env: ConfigType, wrappers: CollectionOfConfigType, agent: ConfigType):
        """Run the parallel rollout in multiple worker processes."""
        workers = self._launch_workers(env, wrappers, agent)
        try:
            self._monitor_rollout(workers)
        except KeyboardInterrupt:
            self._attempt_graceful_exit(workers)

    def _launch_workers(self, env: ConfigType, wrappers: CollectionOfConfigType, agent: ConfigType) \
            -> Iterable[Process]:
        """Configure the workers according to the rollout config and launch them."""
        # Split total episode count across workers
        episodes_per_process = [0] * self.n_processes
        for i in range(self.n_episodes):
            episodes_per_process[i % self.n_processes] += 1

        # Configure and launch the processes
        self.reporting_queue = Queue()
        workers = []
        for n_process_episodes in episodes_per_process:
            if n_process_episodes == 0:
                break

            p = Process(
                target=ParallelRolloutWorker.run,
                args=(env, wrappers, agent,
                      n_process_episodes, self.max_episode_steps,
                      self.record_trajectory, self.input_dir, self.reporting_queue,
                      self.maze_seeding.generate_env_instance_seed(),
                      self.maze_seeding.generate_agent_instance_seed()),
                daemon=True
            )
            p.start()
            workers.append(p)

        # Perform writer registration -- after the forks so that it is not carried over to child processes
        if self.record_event_logs:
            LogEventsWriterRegistry.register_writer(LogEventsWriterTSV(log_dir="./event_logs"))
        register_log_stats_writer(LogStatsWriterConsole())
        self.epoch_stats_aggregator = LogStatsAggregator(LogStatsLevel.EPOCH)
        self.epoch_stats_aggregator.register_consumer(get_stats_logger("rollout_stats"))

        return workers

    def _monitor_rollout(self, workers: Iterable[Process]) -> None:
        """Collect the stats and event logs from the rollout, print progress, and join the workers when done."""

        for _ in tqdm(range(self.n_episodes), desc="Episodes done", unit=" episodes"):
            report = self.reporting_queue.get()
            if isinstance(report, ExceptionReport):
                for p in workers:
                    p.terminate()
                raise RuntimeError("A worker encountered the following error:\n"
                                   + report.traceback) from report.exception

            episode_stats, episode_event_log = report
            if episode_stats is not None:
                self.epoch_stats_aggregator.receive(episode_stats)
            if episode_event_log is not None:
                LogEventsWriterRegistry.record_event_logs(episode_event_log)

        for w in workers:
            w.join()

        if len(self.epoch_stats_aggregator.input) != 0:
            self.epoch_stats_aggregator.reduce()

    def _attempt_graceful_exit(self, workers: Iterable[Process]) -> None:
        """Print statistics collected so far and exit gracefully."""

        print("\n\nShut down requested, exiting gracefully...\n")

        for w in workers:
            w.terminate()

        if len(self.epoch_stats_aggregator.input) != 0:
            print("Stats from the completed part of rollout:\n")
            self.epoch_stats_aggregator.reduce()

        print("\nRollout done (terminated prematurely).")
