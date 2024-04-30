"""Test the creation of data loaders from recorded trajectories."""
import pickle

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from maze.core.agent.random_policy import RandomPolicy
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.trajectory_recording.datasets.in_memory_dataset import InMemoryDataset, get_maze_dataset_class
from maze.core.trajectory_recording.datasets.trajectory_processor import IdentityTrajectoryProcessor
from maze.core.wrappers.time_limit_wrapper import TimeLimitWrapper
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_environment_with_discrete_action_space


def _create_multistep_trajectory():
    """Generates a multistep trajectory."""
    env = build_dummy_maze_environment_with_discrete_action_space(2)
    env = TimeLimitWrapper.wrap(env, max_episode_steps=96)
    rollout_generator = RolloutGenerator(env, record_logits=False, record_step_stats=True,
                                         record_episode_stats=True, record_next_observations=False,
                                         terminate_on_done=True)

    return rollout_generator.rollout(RandomPolicy(env.action_spaces_dict), 1000)


def test_sampling_from_flatten_in_memory_dataset():
    """Tests the FlattenInMemoryDataset and the custom batch sampler."""
    trajectory_name = "trajectory.pkl"
    seed = 1234
    batch_size = 8
    trajectory_record_org = _create_multistep_trajectory()
    with open(trajectory_name, "wb") as out_f:
        pickle.dump(trajectory_record_org, out_f)
    dataset = InMemoryDataset(input_data=trajectory_name, n_workers=0, conversion_env_factory=None,
                              trajectory_processor=IdentityTrajectoryProcessor(),
                              deserialize_in_main_thread=False)
    rng = torch.Generator()
    rng.manual_seed(seed)
    data_loader = get_maze_dataset_class(dataset).create_data_loader(dataset, batch_size=batch_size, num_workers=1,
                                                                     generator=rng)
    iteration = 0
    for iteration, data in enumerate(data_loader, 0):
        observations, actions, actor_ids = data[0], data[1], data[-1]
        assert len(np.unique(actor_ids[0].agent_id)) == 1
    assert (iteration + 1 == len(trajectory_record_org) // batch_size)


def test_sampling_from_split_dataset():
    """Tests the batch sampler and data loader from the split set."""
    trajectory_name = "trajectory.pkl"
    seed = 1234
    validation_size = 10
    batch_size = 8
    trajectory_record_org = _create_multistep_trajectory()
    with open(trajectory_name, "wb") as out_f:
        pickle.dump(trajectory_record_org, out_f)
    dataset = InMemoryDataset(input_data=trajectory_name, n_workers=0, conversion_env_factory=None,
                              trajectory_processor=IdentityTrajectoryProcessor(),
                              deserialize_in_main_thread=False)
    rng = torch.Generator()
    rng.manual_seed(seed)
    validation_set, train_set = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[validation_size, len(dataset) - validation_size],
        generator=torch.Generator().manual_seed(seed))
    data_loader = get_maze_dataset_class(train_set).create_data_loader(train_set, batch_size=batch_size, num_workers=1,
                                                                       generator=rng)
    n_batches_sampled = 0
    for data in data_loader:
        observations, actions, actor_ids = data[0], data[1], data[-1]
        assert len(np.unique(actor_ids[0].agent_id)) == 1
        n_batches_sampled += 1
    assert (n_batches_sampled == len(train_set) // batch_size + int(len(train_set) / batch_size > 0)), \
        f'{n_batches_sampled} != {len(train_set) // batch_size}'


def test_sampling_from_concat_dataset():
    """Tests the batch sampler and data loader from the concatenated sets."""
    trajectory_name = "trajectory-{}-{}.pkl"
    seed = 1234
    batch_size = 8
    datasets = []
    cumulative_trajectories_length = 0
    rng = torch.Generator()
    for idx in range(2):
        trajectory_names = []
        for idx_2 in range(5):
            tmp_trajectory_name = trajectory_name.format(idx, idx_2)
            trajectory_record_org = _create_multistep_trajectory()
            cumulative_trajectories_length += len(trajectory_record_org)
            with open(tmp_trajectory_name, "wb") as out_f:
                pickle.dump(trajectory_record_org, out_f)
            trajectory_names.append(tmp_trajectory_name)
        dataset = InMemoryDataset(input_data=trajectory_names, n_workers=0, conversion_env_factory=None,
                                  trajectory_processor=IdentityTrajectoryProcessor(),
                                  deserialize_in_main_thread=False)
        train, valid = dataset.random_split([int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)], rng)
        datasets.append(train)
    dataset = ConcatDataset(datasets)

    rng.manual_seed(seed)

    data_loader = get_maze_dataset_class(dataset).create_data_loader(dataset, batch_size=batch_size, num_workers=1,
                                                                     generator=rng)
    n_batches_sampled = 0
    for iteration, data in enumerate(data_loader, 0):
        observations, actions, actor_ids = data[0], data[1], data[-1]
        assert len(np.unique(actor_ids[0].agent_id)) == 1
        n_batches_sampled += 1
    assert (n_batches_sampled == len(dataset) // batch_size)
