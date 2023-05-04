"""File holding tests for the random policy."""
import copy
from typing import List

import numpy as np
from gym import spaces

from maze.core.agent.random_policy import RandomPolicy, MaskedRandomPolicy
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.test.shared_test_utils.helper_functions import build_dummy_maze_env, build_dummy_structured_env


def test_default_action_space_sampling():
    env = build_dummy_maze_env()
    policy = RandomPolicy(env.action_spaces_dict)
    action = policy.compute_action(observation=env.observation_space.sample(), maze_state=None)
    assert action in env.action_space


def create_random_action_mask(observation: ObservationType, rng: np.random.RandomState,
                              mask_nothing: bool) -> ObservationType:
    """Create a random action mask.

    :param observation: The observation to edit.
    :parma rng: Numpy random state.
    :param mask_nothing: If true, do not mask anything.
    """
    for observation_key in observation:
        if observation_key.endswith('mask') and not mask_nothing:
            observation[observation_key] = rng.randint(0, 2, size=observation[observation_key].shape)
        elif observation_key.endswith('mask'):
            observation[observation_key] = np.ones(shape=observation[observation_key].shape)

    return observation


def test_masked_random_policy_equality_without_masking():
    """Test for equivalence with the normal random policy."""
    env = build_dummy_maze_env()
    policy = RandomPolicy(env.action_spaces_dict)
    policy_m = MaskedRandomPolicy(copy.deepcopy(env.action_spaces_dict))
    policy.seed(1234)
    policy_m.seed(1234)
    rng = np.random.RandomState(1234)

    obs = env.reset()
    obs = create_random_action_mask(obs, rng, mask_nothing=True)
    for i in range(10):
        action = policy.compute_action(observation=obs, maze_state=None)
        action_m = policy_m.compute_action(observation=obs, maze_state=None)

        obs, rew, done, info = env.step(action)
        obs = create_random_action_mask(obs, rng, mask_nothing=True)
        for key in action.keys():
            assert np.all(np.isclose(action[key], action_m[key]))


def test_masked_random_policy_equality_with_masking():
    """Test for equivalence with the normal random policy."""
    env = build_dummy_maze_env()
    policy = RandomPolicy(env.action_spaces_dict)
    policy_m = MaskedRandomPolicy(copy.deepcopy(env.action_spaces_dict))
    policy.seed(1234)
    policy_m.seed(1234)
    rng = np.random.RandomState(1234)

    obs = env.reset()
    obs = create_random_action_mask(obs, rng, mask_nothing=False)
    all_same = True
    for i in range(10):
        action = policy.compute_action(observation=obs, maze_state=None)
        action_m = policy_m.compute_action(observation=obs, maze_state=None)

        obs, rew, done, info = env.step(action)
        obs = create_random_action_mask(obs, rng, mask_nothing=False)
        for key in action.keys():
            all_same = all_same and np.all(np.isclose(action[key], action_m[key]))
        if not all_same:
            break
    assert not all_same


def check_sampled_action(actions_to_test: List[ActionType], test_env: MazeEnv, observation: ObservationType):
    """Make sure sampled actions are not masked out."""
    for action in actions_to_test:
        for action_key in action.keys():
            if isinstance(test_env.action_space[action_key], spaces.Discrete):
                assert observation[action_key + '_mask'][action[action_key]]
            if isinstance(test_env.action_space[action_key], spaces.MultiBinary):
                assert np.all(observation[action_key + '_mask'][action[action_key].astype(bool)])


def test_masked_random_policy():
    """Test the masked random policy with the structure env."""
    env = build_dummy_structured_env()
    policy_m = MaskedRandomPolicy(copy.deepcopy(env.action_spaces_dict))
    assert not policy_m.needs_state()
    policy_m.seed(1234)
    rng = np.random.RandomState(1234)

    obs = env.reset()
    obs = create_random_action_mask(obs, rng, mask_nothing=False)

    for i in range(10):
        actions, probs = policy_m.compute_top_action_candidates(observation=obs, num_candidates=2, maze_state=None,
                                                                env=None,
                                                                actor_id=env.actor_id())
        check_sampled_action(actions, env, obs)

        obs, rew, done, info = env.step(actions[0])
        obs = create_random_action_mask(obs, rng, mask_nothing=False)
        actions, probs = policy_m.compute_top_action_candidates(observation=obs, num_candidates=2, maze_state=None,
                                                                env=None,
                                                                actor_id=env.actor_id())
        check_sampled_action(actions, env, obs)
        obs, rew, done, info = env.step(actions[0])
        obs = create_random_action_mask(obs, rng, mask_nothing=False)


def test_sampling_without_replacement_single_step():
    action_space_dict = {
        0: spaces.Dict({'action': spaces.Discrete(n=10)})
    }

    pp = RandomPolicy(copy.deepcopy(action_space_dict))

    actions, probs = pp.compute_top_action_candidates(observation={}, num_candidates=None, maze_state=None,
                                                      env=None, actor_id=None)

    assert len(actions) == 10
    action_ids = [elem['action'] for elem in actions]
    assert set(action_ids) == set(range(10))

    actions_2, probs = pp.compute_top_action_candidates(observation={}, num_candidates=None, maze_state=None,
                                                        env=None, actor_id=ActorID(0, 0))

    action_ids_2 = [elem['action'] for elem in actions_2]
    assert set(action_ids_2) == set(action_ids)
    # check if the candidates are permuted
    assert action_ids_2 != action_ids


def test_masked_sampling_without_replacement_structured():
    """Test that if multiple actions are present in the space we can not do sampling without replacement."""
    action_space_dict = {
        0: spaces.Dict({'action': spaces.Discrete(n=100),
                        'action_2': spaces.Discrete(n=10)})
    }

    obs = {
        'action_mask': np.concatenate([np.zeros(50), np.ones(50)])
    }

    pp = MaskedRandomPolicy(copy.deepcopy(action_space_dict))

    actions, probs = pp.compute_top_action_candidates(observation=obs, num_candidates=None, maze_state=None,
                                                      env=None, actor_id=None)

    assert len(actions) == 50 * 10
    action_ids = [elem['action'] for elem in actions]
    action_2_ids = [elem['action_2'] for elem in actions]
    assert set(action_ids) == set(range(50, 100))
    assert set(action_2_ids) == set(range(10))

    actions_2, probs = pp.compute_top_action_candidates(observation=obs, num_candidates=None, maze_state=None,
                                                        env=None, actor_id=ActorID(0, 0))

    action_ids_2 = [elem['action'] for elem in actions_2]
    action_2_ids_2 = [elem['action_2'] for elem in actions_2]
    assert set(action_ids_2) == set(action_ids)
    assert set(action_2_ids_2) == set(action_2_ids)
    # check if the candidates are permuted
    assert action_ids_2 != action_ids
    assert action_2_ids_2 != action_2_ids


def test_sampling_without_replacement_structured():
    """Test that if multiple actions are present in the space we can not do sampling without replacement."""
    action_space_dict = {
        0: spaces.Dict({'action': spaces.Discrete(n=100),
                        'action_2': spaces.Discrete(n=10)})
    }

    pp = RandomPolicy(copy.deepcopy(action_space_dict))

    actions, probs = pp.compute_top_action_candidates(observation={}, num_candidates=None, maze_state=None,
                                                      env=None, actor_id=None)

    assert len(actions) == 100 * 10
    action_ids = [elem['action'] for elem in actions]
    action_2_ids = [elem['action_2'] for elem in actions]
    assert set(action_ids) == set(range(100))
    assert set(action_2_ids) == set(range(10))

    actions_2, probs = pp.compute_top_action_candidates(observation={}, num_candidates=None, maze_state=None,
                                                        env=None, actor_id=ActorID(0, 0))

    action_ids_2 = [elem['action'] for elem in actions_2]
    action_2_ids_2 = [elem['action_2'] for elem in actions_2]
    assert set(action_ids_2) == set(action_ids)
    assert set(action_2_ids_2) == set(action_2_ids)
    # check if the candidates are permuted
    assert action_ids_2 != action_ids
    assert action_2_ids_2 != action_2_ids
