"""
Utils for example notebooks.
"""

import contextlib
import os
import subprocess
import sys
from typing import Union

from maze.api.run_context import RunContext
from maze.core.agent.policy import Policy
from maze.core.env.maze_env import MazeEnv
from maze.core.trajectory_recording.utils.monitoring_setup import MonitoringSetup


def fix_gym_syspath() -> None:
    """
    Imports gym via sys.path. Workaround for gym import not working in notebooks with certain setups.
    """

    try:
        # Try to import gym to see if messing with sys.path is necessary.
        import gym
    except ModuleNotFoundError:
        try:
            # Get info on gym installation path.
            pip_show_output = str(subprocess.check_output(['pip', "show", 'gym']))
            # Extract gym path and append to sys.path.
            sys.path.append(pip_show_output.split("\\n")[-4].split(": ")[1])
        # gym is actually not installed.
        except subprocess.CalledProcessError:
            print("gym is not installed. Please install with: pip install gym.")


def rollout(
    env: MazeEnv, agent: Union[RunContext, Policy], n_max_steps: int, render: bool = False, log_dir: str = "."
) -> float:
    """
    Rolls out environment and collects total reward for one epoch.
    Note: Should be dropped in favour of RunContext.rollout() as soon as possible.

    :param env: Environment to roll out.
    :param agent: Agent with .compute_action(obs).
    :param n_max_steps: Max. number of steps to take.
    :param render: Whether to render the environment.
    :param log_dir: Logging directory.
    :return: Cumulative reward.
    """

    cumulative_reward = 0
    i = 0

    # Suppress output.
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            # Initialize MonitoringSetup for rendering support.
            with MonitoringSetup(env, log_dir=log_dir) as monitored_env:
                obs = monitored_env.reset()
                done = False

                while not done and i < n_max_steps:
                    action = agent.compute_action(obs,
                                                  maze_state=monitored_env.get_maze_state(),
                                                  actor_id=monitored_env.actor_id())
                    obs, reward, done, _ = monitored_env.step(action)
                    cumulative_reward += reward

                    if render:
                        monitored_env.render()

                    i += 1

    env.close()

    return cumulative_reward
