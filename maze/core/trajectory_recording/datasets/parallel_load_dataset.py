"""An imitation dataset that loads all data to memory on initialization using multiple worker processes."""
import logging
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Callable, List, Union, Optional

from tqdm import tqdm

from maze.core.trajectory_recording.datasets.in_memory_dataset import InMemoryDataset
from maze.core.trajectory_recording.datasets.sequential_load_dataset import SequentialLoadDataset
from maze.core.trajectory_recording.records.trajectory_record import TrajectoryRecord
from maze.utils.exception_report import ExceptionReport

logger = logging.getLogger(__name__)


class ParallelLoadDataset(InMemoryDataset):
    """A version of the in-memory dataset that loads all data in parallel.

    This significantly speeds up data loading in cases where conversion of MazeStates and MazeActions into actions
    and observations is demanding.

    :param n_workers: Number of worker processes to load data in.
    :param dir_or_file: Where to load the trajectories from.
                        If this is a file, its contents will be deserialized in the main process, split into
                        chunks and the conversion will then happen in the workers.
                        If this is a directory, individual files in it will be allocated to workers, and hence
                        both deserialization and conversion will happen in the workers.
    :param conversion_env_factory: See the parent class.
    """

    def __init__(self,
                 n_workers: int,
                 dir_or_file: Optional[Union[str, Path]] = None,
                 conversion_env_factory: Optional[Callable] = None):
        self.n_workers = n_workers
        self.reporting_queue = None
        super().__init__(dir_or_file, conversion_env_factory)

    def load_data(self, dir_or_file: Union[str, Path]) -> None:
        """Load the trajectory data based on arguments provided on init."""
        logger.info(f"Started loading trajectory data from: {dir_or_file}")

        if Path(dir_or_file).is_file():
            logger.info(f"Loading single file => deserializing the file in the main process, then loading in workers")
            paths_or_trajectories = [t for t in self.deserialize_trajectories(dir_or_file)]
        else:
            logger.info(f"Loading a directory => deserialization of files split across workers")
            paths_or_trajectories = self.list_trajectory_files(dir_or_file)

        # Split trajectories across workers
        chunks = [[] for _ in range(self.n_workers)]
        for i, trajectory in enumerate(paths_or_trajectories):
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
        progress_bar = tqdm(desc="Loaded", unit=" trajectories")
        n_workers_done = 0
        while n_workers_done < len(workers):
            report = self.reporting_queue.get()

            # Count done workers
            if report == DataLoadWorker.DONE_TOKEN:
                n_workers_done += 1
                continue

            # Report exceptions in the main process
            if isinstance(report, ExceptionReport):
                for p in workers:
                    p.terminate()
                raise RuntimeError(
                    "A worker encountered the following error:\n" + report.traceback) from report.exception

            # Store loaded trajectories
            step_records = report
            self._store_loaded_trajectory(step_records)
            progress_bar.update()

        progress_bar.close()

        for w in workers:
            w.join()

        logger.info(f"Loaded trajectory data from: {dir_or_file}")
        logger.info(f"Current length is {len(self)} steps in total.")


class DataLoadWorker:
    """Data loading worker used to map states to actual observations."""
    DONE_TOKEN = "DONE"

    @staticmethod
    def run(env_factory: Callable,
            trajectories_or_paths: List[Union[Path, str, TrajectoryRecord]],
            reporting_queue: Queue) -> None:
        """Load trajectory data from the provided trajectory file paths. Report exceptions to the main process.

        :param env_factory: Function for creating an environment for MazeState and MazeAction conversion.
        :param trajectories_or_paths: Either file paths to load, or already loaded trajectories to convert.
        :param reporting_queue: Queue for reporting loaded data and exceptions back to the main process.
        """
        try:
            env = env_factory() if env_factory else None
            for trajectory_or_path in trajectories_or_paths:

                # If we got a file path, then deserialize, convert, and report all trajectories in it
                if isinstance(trajectory_or_path, Path) or isinstance(trajectory_or_path, str):
                    for trajectory in SequentialLoadDataset.deserialize_trajectories(trajectory_or_path):
                        step_records = SequentialLoadDataset.convert_trajectory(trajectory, env)
                        reporting_queue.put(step_records)

                # If we got an already-loaded trajectory, then just convert and report it
                elif isinstance(trajectory_or_path, TrajectoryRecord):
                    step_records = SequentialLoadDataset.convert_trajectory(trajectory_or_path, env)
                    reporting_queue.put(step_records)

                else:
                    raise RuntimeError(f"Expected a path or a loaded trajectory record, got {type(trajectory_or_path)}")

            reporting_queue.put(DataLoadWorker.DONE_TOKEN)

        except Exception as exception:
            # Ship exception along with a traceback to the main process
            reporting_queue.put(ExceptionReport(exception))
            raise
