"""General interface for performing and recording policy rollouts during training."""

from typing import Union, Optional, Any

import numpy as np

from maze.core.agent.policy import Policy
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.maze_env import MazeEnv
from maze.core.log_stats.log_stats import LogStatsLevel
from maze.core.trajectory_recording.records.spaces_record import SpacesRecord
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.train.parallelization.vector_env.structured_vector_env import StructuredVectorEnv


class RolloutGenerator:
    """Rollouts a given policy in a given environment, recording the trajectory (in the form of raw actions
    and observations).

    Works with both standard and vectorized environments.

    :param env: Environment to run rollouts in. Will be reset before the first rollout.
    :param record_logits: Whether to record the policy logits.
    :param record_step_stats: Whether to record step statistics.
    :param record_episode_stats: Whether to record episode stats (happens only when an episode is done).
    :param record_next_observations: Whether to record next observation (i.e. observation following the action taken).
    :param terminate_on_done: Whether to end the rollout when the env is done (by default resets the env and continues
                              until the desired number of steps has been recorded). Only applicable in non-vectorized
                              scenarios.
    """

    def __init__(self,
                 env: Union[MazeEnv, StructuredVectorEnv],
                 record_logits: bool = False,
                 record_step_stats: bool = False,
                 record_episode_stats: bool = False,
                 record_next_observations: bool = False,
                 terminate_on_done: bool = False):
        self.env = env
        self.is_vectorized = isinstance(self.env, StructuredVectorEnv)
        self.record_logits = record_logits
        self.record_step_stats = record_step_stats
        self.record_episode_stats = record_episode_stats
        self.record_next_observations = record_next_observations
        self.terminate_on_done = terminate_on_done

        if (self.record_step_stats or self.record_episode_stats) and not isinstance(self.env, LogStatsWrapper):
            self.env = LogStatsWrapper.wrap(self.env)

        self.step_keys = list(env.observation_spaces_dict.keys())  # Only synchronous environments are supported
        self.last_observation = None  # Keep last observations and do not reset envs between rollouts

        self.rollout_counter = 0  # For generating trajectory IDs if none are supplied

    def rollout(self, policy: Policy, n_steps: Optional[int], trajectory_id: Optional[Any] = None) \
            -> SpacesTrajectoryRecord:
        """Perform and record a rollout with given policy, for given steps or until done.

        Note that the env is only reset on the very first rollout with this generator, the following rollouts
        just pick up where the previous left off. If required, you can avoid the initial reset by assigning
        the last observation (which will be recorded with the first step) into `self.last_observation`.

        :param policy: Policy to roll out.
        :param n_steps: How many steps to perform. If None, rollouts are performed until done=True.
        :param trajectory_id: Optionally, the ID of the trajectory that we are recording.
        :return: Recorded trajectory.
        """
        # Check: Logits can be recorded with torch policy only
        if self.record_logits:
            assert isinstance(policy, TorchPolicy), "to collect logits, the policy needs to be a Torch policy"

        # Initialize a trajectory record
        trajectory_record = SpacesTrajectoryRecord(trajectory_id if trajectory_id else self.rollout_counter)
        self.rollout_counter += 1

        # Reset the environment during the first rollout only
        if self.last_observation is None:
            self.last_observation = self.env.reset()

        # Step the desired number of (flat) steps
        step_count = 0
        while True:
            step_record = StructuredSpacesRecord()

            # Step through all sub-steps, i.e., step until the env time changes
            # Note: If the env returns done in a sub-step, this is detected as well as env time changes after reset
            current_env_time = self.env.get_env_time()
            while np.all(current_env_time == self.env.get_env_time()):
                step_record.append(self._record_sub_step(policy=policy))

            if self.record_step_stats:
                step_record.step_stats = self.env.get_stats(LogStatsLevel.STEP).last_stats

            if self.record_episode_stats and not self.is_vectorized and step_record.is_done():
                step_record.episode_stats = self.env.get_stats(LogStatsLevel.EPISODE).last_stats

            # Redistribute actor rewards, if available
            actor_rewards = self.env.get_actor_rewards()
            if actor_rewards is not None:
                assert len(actor_rewards) == len(step_record.substep_records)
                for substep_record, reward in zip(step_record.substep_records, actor_rewards):
                    substep_record.reward = reward

            trajectory_record.append(step_record)

            # Limit maximum number of steps
            step_count += 1
            if n_steps and step_count >= n_steps:
                break

            # End prematurely on env done if desired
            if self.terminate_on_done and not self.is_vectorized and step_record.is_done():
                break

        return trajectory_record

    def _record_sub_step(self, policy: Union[Policy, TorchPolicy]) -> SpacesRecord:
        """Perform one sub-step in the environment and return the record of it.

        Resets non-vectorised envs when done.

        :param policy: The policy to roll out.
        :return: Spaces record with the data recorded during this sub-step.
        """
        record = SpacesRecord(actor_id=self.env.actor_id(),
                              batch_shape=[self.env.n_envs] if self.is_vectorized else None)

        # Record copy of the observation (as by default, the policy converts and handles it in place)
        record.observation = self.last_observation

        # Sample action and record logits if configured
        # Note: Copy the observation (as by default, the policy converts and handles it in place)
        if self.record_logits:
            action, logits = policy.compute_action_with_logits(self.last_observation.copy(), actor_id=record.actor_id,
                                                               deterministic=False)
            record.logits = logits
        else:
            # Inject the MazeEnv state if desired by the policy
            maze_state = self.env.get_maze_state() if policy.needs_state() else None
            env = self.env if policy.needs_env() else None
            action = policy.compute_action(self.last_observation.copy(),
                                           actor_id=record.actor_id,
                                           maze_state=maze_state,
                                           env=env,
                                           deterministic=False)
        record.action = action

        # Take the step
        self.last_observation, record.reward, record.done, record.info = self.env.step(action)

        # Record the resulting observation if requested
        if self.record_next_observations:
            record.next_observation = self.last_observation

        # Reset the env if done and keep the terminal observation
        if not self.is_vectorized and record.done:
            record.info["terminal_observation"] = self.last_observation
            self.last_observation = self.env.reset()

        return record
