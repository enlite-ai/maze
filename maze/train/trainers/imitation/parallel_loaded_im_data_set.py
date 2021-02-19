"""An imitation dataset that loads all data to memory on initialization using multiple worker processes."""

import pickle
import traceback
from collections import namedtuple
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Callable, List, Union

from tqdm import tqdm

from maze.core.trajectory_recorder.episode_record import EpisodeRecord
from maze.train.trainers.imitation.in_memory_data_set import InMemoryImitationDataSet

ActionTuple = namedtuple("ActionTuple", "observations actions")
ExceptionReport = namedtuple("ExceptionReport", "exception traceback")


class DataLoadWorker:
    """Data loading worker used to map states to actual observations."""

    @staticmethod
    def run(env_factory: Callable, trajectory_file_paths: List[Union[Path, str]], reporting_queue: Queue) -> None:
        """Load trajectory data from the provided trajectory file paths. Report exceptions to the main process.

        :param env_factory: Function for creating an environment for MazeState and MazeAction conversion.
        :param trajectory_file_paths: Which trajectory data files should this worker load and process.
        :param reporting_queue: Queue for reporting loaded data and exceptions back to the main process.
        """
        try:
            env = env_factory()
            for trajectory_file_path in trajectory_file_paths:
                with open(str(trajectory_file_path), "rb") as in_f:
                    episode_record: EpisodeRecord = pickle.load(in_f)
                observations, actions = InMemoryImitationDataSet.load_episode_record(env, episode_record)
                reporting_queue.put(ActionTuple(observations, actions))

        except Exception as exception:
            # Ship exception along with a traceback to the main process
            exception_report = ExceptionReport(exception, traceback.format_exc())
            reporting_queue.put(exception_report)
            raise


class ParallelLoadedImitationDataset(InMemoryImitationDataSet):
    """A version of the in-memory dataset that loads all data in parallel.

    This significantly speeds up data loading in cases where conversion of MazeStates and MazeActions into actions
    and observations is demanding.

    :param trajectory_data_dir: See the parent class.
    :param env_factory: See the parent class.
    :param n_workers: Number of worker processes to load data in.
    """

    def __init__(self,
                 trajectory_data_dir: str,
                 env_factory: Callable,
                 n_workers: int):
        self.n_workers = n_workers

        super().__init__(trajectory_data_dir, env_factory)

    def _load_trajectory_data(self) -> None:
        """Load the trajectory data based on arguments provided on init."""
        print(f"Started loading trajectory data...")

        self.observations = []
        self.actions = []

        trajectory_file_paths = self.get_trajectory_files(self.trajectory_data_dir)
        total_trajectories = len(trajectory_file_paths)

        # Split trajectories across workers
        chunks = [[] for _ in range(self.n_workers)]
        for i, trajectory in enumerate(trajectory_file_paths):
            chunks[i % self.n_workers].append(trajectory)

        # Configure and launch the processes
        self.reporting_queue = Queue()
        workers = []
        for trajectories_chunk in chunks:
            if not trajectories_chunk:
                break

            p = Process(
                target=DataLoadWorker.run,
                args=(self.env_factory, trajectories_chunk, self.reporting_queue),
                daemon=True
            )
            p.start()
            workers.append(p)

        # Monitor the loading process
        for _ in tqdm(range(total_trajectories)):
            report = self.reporting_queue.get()

            # Report exceptions in the main process
            if isinstance(report, ExceptionReport):
                for p in workers:
                    p.terminate()
                raise RuntimeError(
                    "A worker encountered the following error:\n" + report.traceback) from report.exception

            observations, actions = report
            self._store_episode_data(observations, actions)

        for w in workers:
            w.join()

        print(f"Loaded trajectory data for {len(self.actions)} steps in total.")
