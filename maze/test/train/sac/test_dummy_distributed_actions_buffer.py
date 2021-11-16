"""Unit test for the SAC implementation"""

import torch
from omegaconf import DictConfig

from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.utils.config_utils import EnvFactory
from maze.core.utils.factory import Factory
from maze.core.wrappers.observation_normalization.observation_normalization_utils import \
    obtain_normalization_statistics, make_normalized_env_factory
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.perception_utils import observation_spaces_to_in_shapes
from maze.test.shared_test_utils.hydra_helper_functions import load_hydra_config
from maze.train.parallelization.distributed_actors.dummy_distributed_workers_with_buffer import \
    DummyDistributedWorkersWithBuffer
from maze.train.trainers.common.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from maze.train.trainers.sac.sac_runners import SACRunner


def create_dummy_distributed_actors(rollouts_per_iteration: int, n_rollout_steps: int,
                                    split_rollouts_into_transitions: bool, n_actors: int):
    """Create a dummy distributed actors and tests the rollout and buffer
    """

    hydra_overrides = {'env': 'gym_env', 'env.name': "LunarLanderContinuous-v2", 'model': 'vector_obs',
                       'model.critic': 'null', 'wrappers': 'vector_obs'}
    cfg = load_hydra_config('maze.conf', 'conf_rollout', hydra_overrides)
    env_factory = EnvFactory(cfg.env, cfg.wrappers if "wrappers" in cfg else {})

    normalization_env = env_factory()
    normalization_statistics = obtain_normalization_statistics(normalization_env,
                                                               n_samples=10)
    if normalization_statistics:
        env_factory = make_normalized_env_factory(env_factory, normalization_statistics)

    model_composer = Factory(base_type=BaseModelComposer).instantiate(
        cfg.model,
        action_spaces_dict=normalization_env.action_spaces_dict,
        observation_spaces_dict=normalization_env.observation_spaces_dict,
        agent_counts_dict=normalization_env.agent_counts_dict)

    initial_sampling_policy = DictConfig({'_target_': 'maze.core.agent.random_policy.RandomPolicy'})

    initial_buffer_size = 30
    batch_size = 20
    replay_buffer_size = 40

    replay_buffer = UniformReplayBuffer(buffer_size=replay_buffer_size, seed=1234)
    SACRunner.init_replay_buffer(replay_buffer=replay_buffer, initial_sampling_policy=initial_sampling_policy,
                                 initial_buffer_size=initial_buffer_size, replay_buffer_seed=1234,
                                 split_rollouts_into_transitions=split_rollouts_into_transitions,
                                 n_rollout_steps=n_rollout_steps, env_factory=env_factory)

    distributed_actors = DummyDistributedWorkersWithBuffer(
        env_factory=env_factory, worker_policy=model_composer.policy, n_rollout_steps=n_rollout_steps,
        n_workers=n_actors, batch_size=batch_size,  rollouts_per_iteration=rollouts_per_iteration,
        split_rollouts_into_transitions=split_rollouts_into_transitions, env_instance_seeds=list(range(n_actors)),
        replay_buffer=replay_buffer)

    print(len(distributed_actors.replay_buffer))
    assert len(distributed_actors.replay_buffer) == initial_buffer_size
    _, _, _ = distributed_actors.collect_rollouts()
    if n_rollout_steps == 1 or not split_rollouts_into_transitions:
        assert len(distributed_actors.replay_buffer) == min(initial_buffer_size + rollouts_per_iteration,
                                                            replay_buffer_size)
    elif split_rollouts_into_transitions:
        assert len(distributed_actors.replay_buffer) == min(initial_buffer_size +
                                                            rollouts_per_iteration * n_rollout_steps,
                                                            replay_buffer_size)
    actor_output = distributed_actors.sample_batch('cpu')
    assert isinstance(actor_output, StructuredSpacesRecord)
    assert hasattr(actor_output, 'observations')
    obs_shapes = observation_spaces_to_in_shapes(normalization_env.observation_spaces_dict)

    for step_key in obs_shapes.keys():
        for obs_key, obs_shape in obs_shapes[step_key].items():
            if not split_rollouts_into_transitions:
                assert actor_output.observations[step_key][obs_key].shape == torch.Size([n_rollout_steps, batch_size,
                                                                                         *obs_shape])
            else:
                assert actor_output.observations[step_key][obs_key].shape == torch.Size([1, batch_size,
                                                                                         *obs_shape])

    return distributed_actors


def test_dummy_distributed_actors_single_rollout_step():
    """ sac unit tests """
    create_dummy_distributed_actors(rollouts_per_iteration=20, n_rollout_steps=1,
                                    split_rollouts_into_transitions=False, n_actors=1)
    create_dummy_distributed_actors(rollouts_per_iteration=20, n_rollout_steps=1,
                                    split_rollouts_into_transitions=True, n_actors=1)

    create_dummy_distributed_actors(rollouts_per_iteration=20, n_rollout_steps=1,
                                    split_rollouts_into_transitions=True, n_actors=2)


def test_dummy_distributed_actors_multiple_rollout_steps():
    """ sac unit tests """
    create_dummy_distributed_actors(rollouts_per_iteration=20, n_rollout_steps=3,
                                    split_rollouts_into_transitions=False, n_actors=1)
    create_dummy_distributed_actors(rollouts_per_iteration=20, n_rollout_steps=3,
                                    split_rollouts_into_transitions=True, n_actors=1)

    create_dummy_distributed_actors(rollouts_per_iteration=20, n_rollout_steps=1,
                                    split_rollouts_into_transitions=True, n_actors=2)
