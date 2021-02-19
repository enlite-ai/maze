"""Trajectory data set for imitation learning."""

import pickle
from itertools import chain
from pathlib import Path
from typing import Callable, Tuple, List, Dict, Union, Any, Sequence

import hydra
import torch
from torch.utils.data.dataset import Dataset, Subset

from maze.core.env.maze_env import MazeEnv
from maze.core.env.structured_env import StructuredEnv
from maze.core.trajectory_recorder.episode_record import EpisodeRecord
from maze.core.wrappers.trajectory_recording_wrapper import RawMazeAction, RawState


class InMemoryImitationDataSet(Dataset):
    """Trajectory data set for imitation learning.

    Loads all data on initialization and then keeps it in memory.
    
    :param trajectory_data_dir: The directory where the trajectory data are stored. 
    :param env_factory: Function for creating an environment for state and action
            conversion. For Maze envs, the environment configuration (i.e. space interfaces,
            wrappers etc.) determines the format of the actions and observations that will be derived
            from the recorded MazeActions and MazeStates (e.g. multi-step observations/actions etc.).
    """

    def __init__(self,
                 trajectory_data_dir: str,
                 env_factory: Callable):
        self.trajectory_data_dir = hydra.utils.to_absolute_path(trajectory_data_dir)
        self.env_factory = env_factory
        self.env = self.env_factory()

        self.observations = []
        self.actions = []
        self.episode_references = []

        self._load_trajectory_data()

    def __len__(self) -> int:
        """Size of the dataset.

        :return: Number of records (i.e. recorded flat-env steps) available
        """
        return len(self.observations)

    def __getitem__(self, index: int) -> Tuple[Dict[Union[int, str], Any], Dict[Union[int, str], Any]]:
        """Get a record.

        :param index: Index of the record to get.
        :return: A tuple of (observation_dict, action_dict). Note that the dictionaries only have multiple entries
                 in structured scenarios.
        """
        return self.observations[index], self.actions[index]

    def _load_trajectory_data(self) -> None:
        """Load the trajectory data based on arguments provided on init."""
        print(f"Started loading trajectory data...")

        self.observations = []
        self.actions = []

        file_paths = self.get_trajectory_files(self.trajectory_data_dir)

        for file_path in file_paths:
            with open(str(file_path), "rb") as in_f:
                episode_record: EpisodeRecord = pickle.load(in_f)
            observations, actions = self.load_episode_record(self.env, episode_record)
            self._store_episode_data(observations, actions)

        print(f"Loaded trajectory data for {len(self.actions)} steps in total.")

    @staticmethod
    def get_trajectory_files(trajectory_data_dir: str) -> List[Path]:
        """List pickle files ("pkl" suffix, used for trajectory data storage by default) in the given directory.

        :param trajectory_data_dir: Where to look for the trajectory records (= pickle files).
        :return: A list of available pkl files in the given directory.
        """
        file_paths = []

        for file_path in Path(trajectory_data_dir).iterdir():
            if file_path.is_file() and file_path.suffix == ".pkl":
                file_paths.append(file_path)

        return file_paths

    @staticmethod
    def load_episode_record(env: StructuredEnv, episode_record: EpisodeRecord) -> \
            Tuple[List[Dict[Union[int, str], Any]], List[Dict[Union[int, str], Any]]]:
        """Convert an episode trajectory record into an array of observations and actions using the given env.

        :param env: Env to use for conversion of MazeStates and MazeActions into observations and actions
        :param episode_record: Episode record to load
        :return: Loaded observations and actions. I.e., a tuple (observation_list, action_list). Each of the
                 lists contains observation/action dictionaries, with keys corresponding to IDs of structured
                 sub-steps. (I.e., the dictionary will have just one entry for non-structured scenarios.)
        """
        observations = []
        actions = []

        for step_id, step_record in enumerate(episode_record.step_records):
            # Drop incomplete records (e.g. at the end of episode)
            if step_record.maze_state is None or step_record.maze_action is None:
                continue

            observation = step_record.maze_state.observation if isinstance(step_record.maze_state,
                                                                           RawState) else step_record.maze_state
            action = step_record.maze_action.action if isinstance(step_record.maze_action,
                                                                  RawMazeAction) else step_record.maze_action

            if isinstance(env, MazeEnv):
                observation, action = env.get_observation_and_action_dicts(observation, action, step_id == 0)

            observations.append(observation)
            actions.append(action)

        return observations, actions

    def _store_episode_data(self,
                            observations: List[Dict[Union[int, str], Any]],
                            actions:List[Dict[Union[int, str], Any]]) -> None:
        """Stores the observations and actions, keeping a reference that they belong to the same episode.

        Keeping the reference is important in case we want to split the dataset later -- samples from
        one episode should end up in the same part (i.e., only training or only validation).
        """
        # Keep a record of which indices belong to the same episode
        offset = len(self.observations)
        self.episode_references.append(range(offset, offset + len(observations)))

        # Store the data
        self.observations.extend(observations)
        self.actions.extend(actions)

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
        shuffled_indices = torch.randperm(len(self.episode_references), generator=generator).tolist()

        # Split episodes across subsets so that each subset has roughly the desired number of step samples
        next_shuffled_idx = 0
        subsets = []

        for desired_subs_len in lengths[:-1]:
            episodes_in_subs = []
            n_samples_in_subs = 0

            # Continue adding episodes into the current subset until we would exceed the desired subset size
            while True:
                next_ep_index = shuffled_indices[next_shuffled_idx]
                next_ep_length = len(self.episode_references[next_ep_index])

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
            sample_indices = list(map(lambda ep_idx: list(self.episode_references[ep_idx]), subset))
            flat_indices = list(chain(*sample_indices))
            data_subsets.append(Subset(self, flat_indices))

        return data_subsets
