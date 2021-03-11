"""An imitation dataset that loads all data to memory on initialization using multiple worker processes."""
import logging
import pickle
import traceback
from collections import namedtuple
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Callable, List, Union, Optional

from tqdm import tqdm

from maze.core.trajectory_recorder.trajectory_record import StateTrajectoryRecord
from maze.train.trainers.imitation.in_memory_data_set import InMemoryImitationDataSet

ExceptionReport = namedtuple("ExceptionReport", "exception traceback")

logger = logging.getLogger(__name__)


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
                    trajectory: StateTrajectoryRecord = pickle.load(in_f)
                step_records = InMemoryImitationDataSet.load_trajectory(trajectory, env)
                reporting_queue.put(step_records)

        except Exception as exception:
            # Ship exception along with a traceback to the main process
            exception_report = ExceptionReport(exception, traceback.format_exc())
            reporting_queue.put(exception_report)
            raise


class ParallelLoadedImitationDataset(InMemoryImitationDataSet):
    """A version of the in-memory dataset that loads all data in parallel.

    This significantly speeds up data loading in cases where conversion of MazeStates and MazeActions into actions
    and observations is demanding.

    :param data_dir: See the parent class.
    :param conversion_env_factory: See the parent class.
    :param n_workers: Number of worker processes to load data in.
    """

    def __init__(self,
                 conversion_env_factory: Callable,
                 n_workers: int,
                 data_dir: Optional[Union[str, Path]] = None):
        self.n_workers = n_workers
        super().__init__(conversion_env_factory, data_dir)

    def load_data(self, data_dir: Union[str, Path]) -> None:
        """Load the trajectory data based on arguments provided on init."""
        logger.info(f"Started loading trajectory data from: {data_dir}")
        file_paths = self.get_trajectory_files(data_dir)
        total_trajectories = len(file_paths)

        # Split trajectories across workers
        chunks = [[] for _ in range(self.n_workers)]
        for i, trajectory in enumerate(file_paths):
            chunks[i % self.n_workers].append(trajectory)

        # Configure and launch the processes
        self.reporting_queue = Queue()
        workers = []
        for trajectories_chunk in chunks:
            if not trajectories_chunk:
                break

            p = Process(
                target=DataLoadWorker.run,
                args=(self.conversion_env_factory, trajectories_chunk, self.reporting_queue),
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

            step_records = report
            self._store_loaded_trajectory(step_records)

        for w in workers:
            w.join()

        logger.info(f"Loaded trajectory data from: {data_dir}")
        logger.info(f"Current length is {len(self)} steps in total.")
