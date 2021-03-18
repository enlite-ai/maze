from typing import Union, Optional, Any

import numpy as np

from maze.core.agent.policy import Policy
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.log_stats.log_stats import LogStatsLevel
from maze.core.trajectory_recording.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.trajectory_record import SpacesTrajectoryRecord
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.train.parallelization.distributed_env.structured_distributed_env import StructuredDistributedEnv


class RolloutGenerator:
    """Rollouts a given policy in a given environment, recording the trajectory (in the form of raw actions
    and observations).

    Works with both standard and distributed environments.

    :param env: Environment to run rollouts in. Will be reset before the first rollout.
    :param record_logits: Whether to record the policy logits.
    :param record_stats:  Whether to record step statistics.
    :param terminate_on_done: Whether to end the rollout when the env is done (by default resets the env and continues
                              until the desired number of steps has been recorded). Only applicable in non-distributed
                              scenarios.
    """
    def __init__(self,
                 env: Union[MazeEnv, StructuredDistributedEnv],
                 record_logits: bool = False,
                 record_stats: bool = False,
                 terminate_on_done: bool = False):
        self.env = env
        self.is_distributed = isinstance(self.env, StructuredDistributedEnv)
        self.record_logits = record_logits
        self.record_stats = record_stats
        self.terminate_on_done = terminate_on_done

        if self.record_stats and not isinstance(self.env, LogStatsWrapper):
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
                                            batch_shape=[self.env.n_envs] if self.is_distributed else None)

            # Step through all sub-steps, i.e., step until the env time changes
            current_env_time = self.env.get_env_time()
            while np.all(current_env_time == self.env.get_env_time()):
                done = self._record_sub_step(record, observation=self.last_observation, policy=policy)

            # Record episode stats
            if self.record_stats:
                record.stats = self.env.get_stats(LogStatsLevel.EPISODE).last_stats

            trajectory_record.append(record)

            # When the env is done in non-distributed scenario, reset or break depending on preferences
            if not self.is_distributed and done:
                if self.terminate_on_done:
                    break
                else:
                    self.env.reset()

        return trajectory_record

    def _record_sub_step(self,
                         record: StructuredSpacesRecord,
                         observation: ObservationType,
                         policy: Union[Policy, TorchPolicy]) -> Union[bool, np.ndarray]:
        """Perform one substep in the environment and record it. Return the done flag(s)."""
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

        return done
