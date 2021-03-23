from typing import Union, Optional, Any

import numpy as np

from maze.core.agent.policy import Policy
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.log_stats.log_stats import LogStatsLevel
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
    :param terminate_on_done: Whether to end the rollout when the env is done (by default resets the env and continues
                              until the desired number of steps has been recorded). Only applicable in non-vectorized
                              scenarios.
    """
    def __init__(self,
                 env: Union[MazeEnv, StructuredVectorEnv],
                 record_logits: bool = False,
                 record_step_stats: bool = False,
                 record_episode_stats: bool = False,
                 terminate_on_done: bool = False):
        self.env = env
        self.is_vectorized = isinstance(self.env, StructuredVectorEnv)
        self.record_logits = record_logits
        self.record_step_stats = record_step_stats
        self.record_episode_stats = record_episode_stats
        self.terminate_on_done = terminate_on_done

        if (self.record_step_stats or self.record_episode_stats) and not isinstance(self.env, LogStatsWrapper):
            self.env = LogStatsWrapper.wrap(self.env)

        self.step_keys = list(env.observation_spaces_dict.keys())  # Only synchronous environments are supported
        self.last_observation = None  # Keep last observations and do not reset envs between rollouts

        self.rollout_counter = 0  # For generating trajectory IDs if none are supplied

    def rollout(self, policy: Policy, n_steps: int, trajectory_id: Optional[Any] = None) -> SpacesTrajectoryRecord:
        """Perform and record a rollout with given policy, for given steps.

        Note that the env is only reset on the very first rollout with this generator, the following rollouts
        just pick up where the previous left off. If required, you can avoid the initial reset by assigning
        the last observation (which will be recorded with the first step) into `self.last_observation`.

        :param policy: Policy to roll out.
        :param n_steps: How many steps to perform.
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
        done = False
        for _ in range(n_steps):
            record = StructuredSpacesRecord(observations={}, actions={}, rewards={}, dones={}, infos={},
                                            logits={} if self.record_logits else None,
                                            batch_shape=[self.env.n_envs] if self.is_vectorized else None)

            # Step through all sub-steps, i.e., step until the env time changes
            # Note: If the env returns done in a sub-step, this is detected as well as env time changes after reset
            current_env_time = self.env.get_env_time()
            while np.all(current_env_time == self.env.get_env_time()):
                done = self._record_sub_step(record, observation=self.last_observation, policy=policy)

            # Record episode stats
            if self.record_step_stats:
                record.step_stats = self.env.get_stats(LogStatsLevel.STEP).last_stats

            trajectory_record.append(record)

            # End prematurely on env done if desired
            if not self.is_vectorized and done and self.terminate_on_done:
                break

        return trajectory_record

    def _record_sub_step(self,
                         record: StructuredSpacesRecord,
                         observation: ObservationType,
                         policy: Union[Policy, TorchPolicy]) -> Union[bool, np.ndarray]:
        """Perform one substep in the environment and record it. Return the done flag(s).

        Resets non-vectorised envs when done.
        """
        step_key, _ = self.env.actor_id()
        # Record copy of the observation (as by default, the policy converts and handles it in place)
        record.observations[step_key] = self.last_observation.copy()

        # Sample action and record logits if configured
        if self.record_logits:
            action, logits = policy.compute_action_with_logits(observation, policy_id=step_key,
                                                               deterministic=False)
            record.logits[step_key] = logits
        else:
            # Inject the MazeEnv state if desired by the policy
            maze_state = self.env.get_maze_state() if policy.needs_state() else None
            action = policy.compute_action(observation, policy_id=step_key, maze_state=maze_state,
                                           deterministic=False)
        record.actions[step_key] = action

        # Take the step
        self.last_observation, reward, done, info = self.env.step(action)

        record.rewards[step_key] = reward
        record.dones[step_key] = done
        record.infos[step_key] = info

        # Reset the env if done and keep the terminal observation
        if not self.is_vectorized and done:
            record.infos[step_key]["terminal_observation"] = self.last_observation
            self.last_observation = self.env.reset()

            if self.record_episode_stats:
                record.episode_stats = self.env.get_stats(LogStatsLevel.EPISODE).last_stats

        return done
