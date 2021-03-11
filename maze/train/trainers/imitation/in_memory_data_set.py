"""Trajectory data set for imitation learning."""
import logging
import pickle
from itertools import chain
from pathlib import Path
from typing import Callable, Tuple, List, Dict, Union, Any, Sequence, Optional, Generator

import torch
from torch.utils.data.dataset import Dataset, Subset

from maze.core.env.structured_env import StructuredEnv
from maze.core.trajectory_recorder.spaces_step_record import SpacesStepRecord
from maze.core.trajectory_recorder.state_step_record import StateStepRecord
from maze.core.trajectory_recorder.trajectory_record import StateTrajectoryRecord, TrajectoryRecord

logger = logging.getLogger(__name__)


class InMemoryImitationDataSet(Dataset):
    """Trajectory data set for imitation learning.

    Loads all data on initialization and then keeps it in memory.
    
    :param data_dir: The directory where the trajectory data are stored. 
    :param conversion_env_factory: Function for creating an environment for state and action
            conversion. For Maze envs, the environment configuration (i.e. space interfaces,
            wrappers etc.) determines the format of the actions and observations that will be derived
            from the recorded MazeActions and MazeStates (e.g. multi-step observations/actions etc.).
    """

    def __init__(self,
                 conversion_env_factory: Callable,
                 dir_or_file: Optional[Union[str, Path]] = None):
        self.conversion_env_factory = conversion_env_factory
        self.conversion_env = self.conversion_env_factory()

        self.step_records = []
        self.trajectory_references = []

        if dir_or_file is not None:
            self.load_data(dir_or_file)

    def __len__(self) -> int:
        """Size of the dataset.

        :return: Number of records (i.e. recorded flat-env steps) available
        """
        return len(self.step_records)

    def __getitem__(self, index: int) -> Tuple[Dict[Union[int, str], Any], Dict[Union[int, str], Any]]:
        """Get a record.

        :param index: Index of the record to get.
        :return: A tuple of (observation_dict, action_dict). Note that the dictionaries only have multiple entries
                 in structured scenarios.
        """
        return self.step_records[index].observations, self.step_records[index].actions

    def append(self, trajectory: TrajectoryRecord):
        spaces_records = self.convert_trajectory(trajectory, self.conversion_env)
        self._store_loaded_trajectory(spaces_records)
        return self

    def load_data(self, dir_or_file: Union[str, Path]) -> None:
        """Load the trajectory data based on arguments provided on init."""
        logger.info(f"Started loading trajectory data from: {dir_or_file}")

        for trajectory in self.deserialize_trajectories(dir_or_file):
            self.append(trajectory)

        logger.info(f"Loaded trajectory data from: {dir_or_file}")
        logger.info(f"Current length is {len(self)} steps in total.")

    @staticmethod
    def deserialize_trajectories(dir_or_file: Union[str, Path]) -> Generator[StateTrajectoryRecord, None, None]:
        dir_or_file = Path(dir_or_file)
        if dir_or_file.is_dir():
            file_paths = InMemoryImitationDataSet.list_trajectory_files(dir_or_file)
        else:
            file_paths = [dir_or_file]

        for file_path in file_paths:
            with open(str(file_path), "rb") as in_f:
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

    @staticmethod
    def convert_trajectory(trajectory: TrajectoryRecord, conversion_env: StructuredEnv) -> List[SpacesStepRecord]:
        """Convert an episode trajectory record into an array of observations and actions using the given env.

        :param env: Env to use for conversion of MazeStates and MazeActions into observations and actions
        :param trajectory: Episode record to load
        :return: Loaded observations and actions. I.e., a tuple (observation_list, action_list). Each of the
                 lists contains observation/action dictionaries, with keys corresponding to IDs of structured
                 sub-steps. (I.e., the dictionary will have just one entry for non-structured scenarios.)
        """
        step_records = []

        for step_id, step_record in enumerate(trajectory.step_records):

            # Process and convert in case we are dealing with state records (otherwise no conversion needed)
            if isinstance(step_record, StateStepRecord):
                # Drop incomplete records (e.g. at the end of episode)
                if step_record.maze_state is None or step_record.maze_action is None:
                    continue
                # Convert to spaces
                step_record = SpacesStepRecord.converted_from(step_record, conversion_env=conversion_env,
                                                              first_step_in_episode=step_id == 0)

            step_records.append(step_record)

        return step_records

    def _store_loaded_trajectory(self, records: List[SpacesStepRecord]) -> None:
        """Stores the observations and actions, keeping a reference that they belong to the same episode.

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
