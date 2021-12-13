"""
Auxiliary routines for reproducibility tests.
"""

import base64
import hashlib
from typing import Any, Tuple, Callable, List, Optional

import numpy as np
from maze.core.agent.policy import Policy

from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType


def hash_deterministically(obj: Any) -> str:
    """
    Produces hash for hashable object wo/ python's .hash() to allow for reproducibility with strings.
    Source: https://stackoverflow.com/a/42151923.
    :param obj: Hashable object to be hashed.
    :return: Hash for obj.
    """

    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(obj)).encode())
    return base64.b64encode(hasher.digest()).decode()


def make_hashable(obj: Any) -> Tuple:
    """
    Converts object (also nested dicts) into tuples to make them hashable.
    Source: https://stackoverflow.com/a/42151923.
    :param obj: Object to make hashable (e.g. nested dictionary).
    :return: Object as hashable tuple.
    """

    if isinstance(obj, (tuple, list)):
        return tuple((make_hashable(e) for e in obj))

    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))

    if isinstance(obj, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in obj))

    if isinstance(obj, np.ndarray):
        return make_hashable(obj.tolist())

    return obj


def conduct_env_reproducibility_test(env: MazeEnv, pick_action: Callable, n_steps: int = 100) -> bool:
    """
    Runs specified environment twice with specified callback to pick action.
    The observations of these two runs have to match.

    :param env: Initialized MazeEnv instance.
    :param pick_action: Reference to function choosing which action to take.
    :param n_steps: Number of steps to run.
    :return: True if reproducible; else False.
    """

    observation_hashes = []
    for _ in range(2):
        observations = run_reproducible_rollout(env, pick_action, n_steps)
        observation_hashes.append(str(hash_deterministically(observations)))

    assert len(observation_hashes) == 2
    return observation_hashes[0] == observation_hashes[1]


def run_reproducible_rollout(env: MazeEnv, pick_action: Callable, n_steps: int = 100) -> List[ObservationType]:
    """
    Runs specified environment with specified callback to pick action.

    :param env: Initialized MazeEnv instance.
    :param pick_action: Reference to function choosing which action to take.
    :param n_steps: Number of steps to run.
    :return: A list of observations collected along the way.
    """
    env.seed(1234)

    # seed the ActionConversion spaces
    act_conv_spaces = dict()
    for policy_id, policy_act_conv in env.action_conversion_dict.items():
        # Get transformation spaces.
        policy_space = policy_act_conv.space()
        # Set randomization seed.
        policy_space.seed(1234)

        act_conv_spaces[policy_id] = policy_space

    # Store hashed step states.
    observations: List[ObservationType] = list()

    env.reset()
    for step in range(n_steps):
        policy_id, actor_id = env.actor_id()

        # Select next action.
        action = pick_action(
            observation=env.observation_conversion.maze_to_space(env.core_env.get_maze_state()),
            action_space=act_conv_spaces[policy_id]
        )

        # Execute action, collect state information.
        observations.append(env.step(action)[0])

    return observations
