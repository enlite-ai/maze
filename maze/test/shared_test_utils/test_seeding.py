"""Helper method to test the seeding behaviour of maze envs"""
from typing import List, Tuple

import numpy as np

from maze.core.agent.policy import Policy
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.utils.seeding import MazeSeeding
from maze.test.shared_test_utils.reproducibility import hash_deterministically


def get_obs_action_hash_for_env_agent(env: MazeEnv, policy: Policy, env_seed: int, agent_seed: int, n_steps: int) \
        -> Tuple[str, str]:
    """Seed the given env and policy with the given seeds and perform a rollout for some steps. Then has the collected
    observations and actions and return the str of the hash keys.

    :param env: The env to perform the rollout on.
    :param policy: The policy to compute the actions.
    :param env_seed: The seed for the env.
    :param agent_seed: The seed for the policy.
    :param n_steps: The number of steps to collect observations and actions.

    :return: The str of the hash of the collected observation and actions.
    """

    env.seed(env_seed)
    policy.seed(agent_seed)

    observations: List[ObservationType] = list()
    actions: List[ActionType] = list()

    obs = env.reset()
    observations.append(obs)
    for step in range(n_steps):
        actor_id = env.actor_id()

        maze_state = env.get_maze_state() if policy.needs_state() else None
        action = policy.compute_action(obs, actor_id=actor_id, maze_state=maze_state, deterministic=False, env=None)

        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

        observations.append(obs)
        actions.append(action)

    return str(hash_deterministically(observations)), str(hash_deterministically(actions))


def perform_seeding_test(env: MazeEnv, policy: Policy, is_deterministic_env: bool, is_deterministic_agent: bool,
                         n_steps: int = 100) \
        -> None:
    """Perform a test on the seeding capabilities of a given env and agent.
        Within this method a rollout is generated with sampled seeds where the observation and actions are recorded and
        hashed. Then a second rollouts is generated with the SAME seeds to check if the results stay the same. Finally
        the seeds are changes and it is checks if the resulting observations and actions change as well to ensure
        randomness in the env.

    :param env: The Env to test.
    :param policy: The policy to compute the actions with.
    :param is_deterministic_env: Specify whether the given env is deterministic.
    :param is_deterministic_agent: Specify whether the given policy is deterministic.
    :param n_steps: Number of steps to perform the comparison on.
    """

    maze_rng = np.random.RandomState(1234)

    agent_seed = MazeSeeding.generate_seed_from_random_state(maze_rng)
    env_seed = MazeSeeding.generate_seed_from_random_state(maze_rng)

    # Perform a rollout and get hash values to compare other runs to.
    base_obs_hash, base_action_hash = get_obs_action_hash_for_env_agent(env, policy, env_seed, agent_seed, n_steps)

    # Perform a second rollout with the same seeds, and check the results are exactly the same.
    second_obs_hash, second_action_hash = get_obs_action_hash_for_env_agent(env, policy, env_seed, agent_seed, n_steps)

    assert base_obs_hash == second_obs_hash
    assert base_action_hash == second_action_hash

    # Change the agent seed and check that the values change if the agent is not deterministic and stay the same if it
    #  is
    agent_seed_2 = MazeSeeding.generate_seed_from_random_state(maze_rng)
    agent_2_obs_hash, agent_2_action_hash = get_obs_action_hash_for_env_agent(env, policy, env_seed, agent_seed_2,
                                                                              n_steps)
    if is_deterministic_agent:
        assert base_obs_hash == agent_2_obs_hash
        assert base_action_hash == agent_2_action_hash
    else:
        assert base_action_hash != agent_2_action_hash

    # Change the env seed and check that the values change if the env is deterministic, and stay the same if it is.
    env_seed_2 = MazeSeeding.generate_seed_from_random_state(maze_rng)
    env_2_obs_hash, env_2_action_hash = get_obs_action_hash_for_env_agent(env, policy, env_seed_2, agent_seed, n_steps)
    if is_deterministic_env:
        assert base_obs_hash == env_2_obs_hash
        assert base_action_hash == env_2_action_hash
    else:
        assert base_obs_hash != env_2_obs_hash

    env_agent_2_obs_hash, env_agent_2_action_hash = get_obs_action_hash_for_env_agent(env, policy, env_seed_2,
                                                                                      agent_seed_2, n_steps)
    if is_deterministic_env and is_deterministic_agent:
        assert base_obs_hash == env_agent_2_obs_hash
        assert base_action_hash == env_agent_2_action_hash
    elif is_deterministic_env:
        assert base_action_hash != env_agent_2_action_hash
    elif is_deterministic_agent:
        assert base_obs_hash != env_agent_2_obs_hash
    else:
        assert base_action_hash != env_agent_2_action_hash
        assert base_obs_hash != env_agent_2_obs_hash
