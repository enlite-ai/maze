from typing import Union, List, Optional, Any

from maze.core.agent.policy import Policy
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats import LogStatsLevel
from maze.core.trajectory_recorder.spaces_step_record import SpacesStepRecord
from maze.core.trajectory_recorder.trajectory_record import SpacesTrajectoryRecord
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.train.parallelization.distributed_env.distributed_env import BaseDistributedEnv
from maze.train.utils.train_utils import unstack_numpy_list_dict


class RolloutGenerator:
    def __init__(self,
                 env: Union[BaseDistributedEnv, StructuredEnv, StructuredEnvSpacesMixin],
                 record_logits: bool = False,
                 record_stats: bool = False,
                 terminate_on_done: bool = False):
        self.env = env
        self.is_distributed = isinstance(self.env, BaseDistributedEnv)
        self.record_logits = record_logits
        self.record_stats = record_stats
        self.terminate_on_done = terminate_on_done

        if self.record_stats and not isinstance(self.env, LogStatsWrapper):
            self.env = LogStatsWrapper.wrap(self.env)

        self.step_keys = list(env.observation_spaces_dict.keys())  # Only synchronous environments are supported
        self.last_observation = None  # Keep last observations and do not reset envs between rollouts

        self.rollout_counter = 0  # For generating trajectory IDs if none are supplied

    def rollout(self, policy: Policy, n_steps: int, trajectory_id: Optional[Any] = None) -> SpacesTrajectoryRecord:
        if self.record_logits:
            assert isinstance(policy, TorchPolicy), "to collect logits, the policy needs to be a Torch policy"

        trajectory_record = SpacesTrajectoryRecord(trajectory_id if trajectory_id else self.rollout_counter)
        self.rollout_counter += 1

        # Reset during the first rollout only
        observation = self.last_observation if self.last_observation else self.env.reset()

        for _ in range(n_steps):
            record = SpacesStepRecord(observations={}, actions={}, rewards={}, dones={}, infos={},
                                      logits={} if self.record_logits else None,
                                      batch_shape=[self.env.n_envs] if self.is_distributed else None)

            for step_key in self.step_keys:
                # Record copy of the observation (as by default, the policy converts and handles it in place)
                record.observations[step_key] = observation.copy()

                # Sample action and record logits if configured
                if self.record_logits:
                    action, logits = policy.compute_action_with_logits(observation, policy_id=step_key,
                                                                       deterministic=False)
                    record.logits[step_key] = logits
                else:
                    action = policy.compute_action(observation, policy_id=step_key, maze_state=None,
                                                   deterministic=False)
                record.actions[step_key] = action

                # Unstack action in distributed env scenarios (the env should handle this in the future)
                if self.is_distributed:
                    action = unstack_numpy_list_dict(action)

                # Take the step
                observation, reward, done, info = self.env.step(action)

                record.rewards[step_key] = reward
                record.dones[step_key] = done
                record.infos[step_key] = info

            if self.record_stats:
                record.stats = self.env.get_stats(LogStatsLevel.EPISODE).last_stats

            trajectory_record.append(record)

            if not self.is_distributed and done:
                if self.terminate_on_done:
                    break
                else:
                    self.env.reset()

        return trajectory_record
