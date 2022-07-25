"""Trajectory data set for imitation learning."""
import itertools
import logging
import pickle
from abc import ABC
from itertools import chain
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Callable, List, Union, Optional, Tuple
from typing import Dict, Sequence, Generator

import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from maze.core.env.action_conversion import ActionType, TorchActionType
from maze.core.env.observation_conversion import ObservationType, TorchObservationType
from maze.core.env.structured_env import ActorID
from maze.core.trajectory_recording.datasets.trajectory_processor import \
    TrajectoryProcessor
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.records.trajectory_record import TrajectoryRecord
from maze.core.utils.factory import ConfigType, Factory
from maze.utils.exception_report import ExceptionReport

logger = logging.getLogger(__name__)


class InMemoryDataset(Dataset, ABC):
    """Base class of trajectory data set for imitation learning that keeps all loaded data in memory.

    Provides the main functionality for parsing and appending records.

    :param input_data: The optional input data to fill the dataset with. This can be either a single file, a single
            directory, a list of files or a list of directories.
    :param conversion_env_factory: Function for creating an environment for state and action
            conversion. For Maze envs, the environment configuration (i.e. space interfaces,
            wrappers etc.) determines the format of the actions and observations that will be derived
            from the recorded MazeActions and MazeStates (e.g. multi-step observations/actions etc.).
    :param n_workers: Number of worker processes to load data in.
    :param trajectory_processor: The processor object for processing and converting individual trajectories.
    :param deserialize_in_main_thread: Specify whether to deserialize the trajectories in the main thread (True) or in
            in the workers. In case only one trajectory file is given and this file holds many trajectories which all
            have to be converted with the conversion env setting this value to true makes sense as the expensive
            operation is the conversion. However if many files are given where no conversion is necessary the expensive
            operation is the deserialization, and thus should happen in the worker threads. Only relevant if
            n_workers > 1.
    """

    def __init__(self,
                 input_data: Optional[Union[str, Path, List[Union[str, Path]]]],
                 conversion_env_factory: Optional[Callable], n_workers: int,
                 trajectory_processor: Union[TrajectoryProcessor, ConfigType], deserialize_in_main_thread: bool):

        self._conversion_env_factory = conversion_env_factory
        self._conversion_env = self._conversion_env_factory() if self._conversion_env_factory else None
        self.n_workers = n_workers
        self._trajectory_processor = Factory(TrajectoryProcessor).instantiate(trajectory_processor)
        self._deserialize_in_main_thread = deserialize_in_main_thread

        self.step_records = []
        self.trajectory_references = []
        self.reporting_queue = None

        if input_data is not None:
            self.load_data(input_data)

    def load_data(self, input_data: Union[str, Path, List[Union[str, Path]]]) -> None:
        """Load the trajectory data from the given file or directory and append it to the dataset.

        Should provide the main logic of how the data load is done to be efficient for the data at hand
        (e.g. splitting it up into multiple parallel workers). Otherwise, this class already provides multiple
        helper methods useful for loading (e.g. for deserializing different structured of trajectories
        or converting maze states to raw observations).

        :param input_data: Input data to load the trajectories from. This can be either a single file, a single
                directory, a list of files or a list of directories.
        """
        if self.n_workers <= 1:
            self._load_data_sequential(input_data)
        else:
            self._load_data_parallel(input_data)

    @classmethod
    def _read_input_data_to_list(cls, input_data: Union[str, Path, List[Union[str, Path]]]) -> List[str]:
        """Read the input data: either a directory, a list of files, a list of dirs or a single file to a list of files.

        :param input_data: The input data.
        :return: A list of files to be loaded.
        """
        if isinstance(input_data, (list, ListConfig)):
            if Path(input_data[0]).is_file():
                trajectory_save_paths = list(input_data)
            else:
                trajectory_save_paths = list()
                for dir_path in input_data:
                    assert Path(dir_path).is_dir()
                    trajectory_save_paths.extend(cls.list_trajectory_files(dir_path))
        elif Path(input_data).is_file():
            trajectory_save_paths = [input_data]
        elif Path(input_data).is_dir():
            trajectory_save_paths = cls.list_trajectory_files(input_data)
        else:
            raise ValueError(f'Unsupported type of data given: {type(input_data)}')

        return trajectory_save_paths

    def _load_data_sequential(self, input_data: Union[Union[str, Path], List[Union[str, Path]]]) -> None:
        """Load data in a sequential fashion."""
        logger.info(f"Started loading trajectory data from: {input_data}")

        trajectory_save_paths = self._read_input_data_to_list(input_data=input_data)

        for file_path in tqdm(trajectory_save_paths, desc='Files', position=0):
            for trajectory in self.deserialize_trajectory(file_path):
                self.append(trajectory)

        logger.info(f"Loaded trajectory data from: {input_data}")
        logger.info(f"Current length is {len(self)} steps in total.")

    def _load_data_parallel(self, dir_or_file: Union[str, Path]) -> None:
        """Load data in a parallel fashion."""
        logger.info(f"Started loading trajectory data from: {dir_or_file}")

        trajectory_save_paths = self._read_input_data_to_list(input_data=dir_or_file)
        if self._deserialize_in_main_thread:
            trajectories_or_paths = list(itertools.chain(*map(self.deserialize_trajectory, trajectory_save_paths)))
        else:
            trajectories_or_paths = trajectory_save_paths

        # Split trajectories across workers
        chunks = [[] for _ in range(self.n_workers)]
        for i, trajectory_or_path in enumerate(trajectories_or_paths):
            chunks[i % self.n_workers].append(trajectory_or_path)

        # Configure and launch the processes
        self.reporting_queue = Queue()
        workers = []
        for trajectories_chunk in chunks:
            if not trajectories_chunk:
                break

            p = Process(
                target=DataLoadWorker.run,
                args=(self._conversion_env_factory, trajectories_chunk, self.reporting_queue,
                      self._trajectory_processor),
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

    def __len__(self) -> int:
        """Size of the dataset.

        :return: Number of records (i.e. recorded flat-env steps) available
        """
        return len(self.step_records)

    def __getitem__(self, index: int) -> Tuple[List[Union[ObservationType, TorchObservationType]],
                                               List[Union[ActionType, TorchActionType]], List[ActorID]]:
        """Get a record.

        :param index: Index of the record to get.
        :return: A tuple of (observations, actions and actor_ids) each as lists corresponding
            to the sub-step of the env (actor_id and step_id).
        """

        return self.step_records[index].observations, self.step_records[index].actions, \
               self.step_records[index].actor_ids

    def append(self, trajectory: TrajectoryRecord) -> None:
        """Append a new trajectory to the dataset.

        :param trajectory: Trajectory to append.
        """
        spaces_records_list = self._trajectory_processor.process(trajectory, self._conversion_env)
        for spaces_record in spaces_records_list:
            self._store_loaded_trajectory(spaces_record)

    @staticmethod
    def deserialize_trajectory(trajectory_file: Union[str, Path]) -> Generator[TrajectoryRecord, None, None]:
        """Deserialize all trajectories located in the given file path.

        Will attempt to load the given trajectory file. Supports pickled TrajectoryRecords, or lists or
        dictionaries containing TrajectoryRecords as values.

        Returns a generator that will yield the individual trajectory records, no matter in which form
        (i.e., individual, list, or dict) they were loaded.

        :param trajectory_file: File to load trajectory data from.
        :return: Generator yielding the individual trajectory records.
        """
        trajectory_file = Path(trajectory_file)
        assert trajectory_file.is_file()

        with open(str(trajectory_file), "rb") as in_f:
            record = pickle.load(in_f)

        # Loading trajectory record directly
        if isinstance(record, TrajectoryRecord):
            yield record

        # Loading a list of trajectory records
        elif isinstance(record, List):
            for item in record:
                assert isinstance(item, TrajectoryRecord)
                yield item

        # Loading a dict of trajectory records
        elif isinstance(record, Dict):
            for item in record.values():
                assert isinstance(item, TrajectoryRecord)
                yield item

        else:
            raise RuntimeError("Unsupported data type, expected a TrajectoryRecord, or list or dict thereof")

    @staticmethod
    def list_trajectory_files(data_dir: Union[str, Path]) -> List[Path]:
        """List pickle files ("pkl" suffix, used for trajectory data storage by default) in the given directory.

        :param data_dir: Where to look for the trajectory records (= pickle files).
        :return: A list of available pkl files in the given directory.
        """
        file_paths = []

        for file_path in Path(data_dir).iterdir():
            if file_path.is_file() and file_path.suffix == ".pkl":
                file_paths.append(file_path)

        return file_paths

    def _store_loaded_trajectory(self, records: List[StructuredSpacesRecord]) -> None:
        """Stores the step records, keeping a reference that they belong to the same episode.

        Keeping the reference is important in case we want to split the dataset later -- samples from
        one episode should end up in the same part (i.e., only training or only validation).
        """
        # Keep a record of which indices belong to the same episode
        offset = len(self)
        self.trajectory_references.append(range(offset, offset + len(records)))

        # Store the data
        self.step_records.extend(records)

    def random_split(self, lengths: Sequence[int], generator: torch.Generator = torch.default_generator) \
            -> List[Subset]:
        """Randomly split the dataset into non-overlapping new datasets of given lengths.

        The split is based on episodes -- samples from the same episode will end up in the same subset. Based
        on the available episode lengths, this might result in subsets of slightly different lengths than specified.

        Optionally fix the generator for reproducible results, e.g.:

            self.random_split([3, 7], generator=torch.Generator().manual_seed(42))

        :param lengths: lengths of splits to be produced (best effort, the result might differ based
                        on available episode lengths
        :param generator: Generator used for the random permutation.
        :return: A list of the data subsets, each with size roughly (!) corresponding to what was specified by lengths.
        """
        if sum(lengths) != len(self):
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        # Shuffle episode indexes, we will then draw them in the new order
        shuffled_indices = torch.randperm(len(self.trajectory_references), generator=generator).tolist()

        # Split episodes across subsets so that each subset has roughly the desired number of step samples
        next_shuffled_idx = 0
        subsets = []
        for desired_subs_len in lengths[:-1]:
            if desired_subs_len == 0:
                continue
            episodes_in_subs = []
            n_samples_in_subs = 0

            # Continue adding episodes into the current subset until we would exceed the desired subset size
            while True:
                next_ep_index = shuffled_indices[next_shuffled_idx]
                next_ep_length = len(self.trajectory_references[next_ep_index])

                if n_samples_in_subs + next_ep_length > desired_subs_len:
                    break

                episodes_in_subs.append(next_ep_index)
                n_samples_in_subs += next_ep_length
                next_shuffled_idx += 1

            subsets.append(episodes_in_subs)

        # Whatever remained ends up in the last subset
        episodes_in_last_subs = shuffled_indices[next_shuffled_idx:]
        subsets.append(episodes_in_last_subs)

        # Get indices of individual samples of each episode and flatten them into a dataset subset
        data_subsets = []
        for subset in subsets:
            sample_indices = list(map(lambda ep_idx: list(self.trajectory_references[ep_idx]), subset))
            flat_indices = list(chain(*sample_indices))
            data_subsets.append(Subset(self, flat_indices))

        return data_subsets


class DataLoadWorker:
    """Data loading worker used to map states to actual observations."""
    DONE_TOKEN = "DONE"

    @staticmethod
    def run(env_factory: Callable,
            trajectories_or_paths: List[Union[Path, str, TrajectoryRecord]],
            reporting_queue: Queue, trajectory_processor: TrajectoryProcessor) -> None:
        """Load trajectory data from the provided trajectory file paths. Report exceptions to the main process.

        :param env_factory: Function for creating an environment for MazeState and MazeAction conversion.
        :param trajectories_or_paths: Either file paths to load, or already loaded trajectories to convert.
        :param reporting_queue: Queue for reporting loaded data and exceptions back to the main process.
        :param trajectory_processor: A trajectory processor class.
        """
        try:
            env = env_factory() if env_factory else None
            for trajectory_or_file in trajectories_or_paths:

                # If we got a file path, then deserialize, convert, and report all trajectories in it
                if isinstance(trajectory_or_file, Path) or isinstance(trajectory_or_file, str):
                    for trajectory in InMemoryDataset.deserialize_trajectory(trajectory_or_file):
                        for step_record in trajectory_processor.process(trajectory, env):
                            reporting_queue.put(step_record)
                # If we got an already-loaded trajectory, then just convert and report it
                elif isinstance(trajectory_or_file, TrajectoryRecord):
                    for step_record in trajectory_processor.process(trajectory_or_file, env):
                        reporting_queue.put(step_record)
                else:
                    raise RuntimeError(f"Expected a path or a loaded trajectory record, got {type(trajectory_or_file)}")

            reporting_queue.put(DataLoadWorker.DONE_TOKEN)

        except Exception as exception:
            # Ship exception along with a traceback to the main process
            reporting_queue.put(ExceptionReport(exception))
            raise
