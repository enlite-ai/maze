from typing import Union, List

from maze.core.agent.policy import Policy
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.trajectory_recorder.spaces_step_record import SpacesStepRecord
from maze.train.parallelization.distributed_env.distributed_env import BaseDistributedEnv
from maze.train.utils.train_utils import unstack_numpy_list_dict


class DistributedEnvRolloutGenerator:
    def __init__(self, env: Union[BaseDistributedEnv, StructuredEnv, StructuredEnvSpacesMixin]):
        self.env = env
        self.step_keys = list(env.observation_spaces_dict.keys())  # Only synchronous environments are supported
        self.last_observations = None  # Keep last observations and do not reset envs between rollouts

    def rollout(self, policy: Policy, n_steps: int) -> List[SpacesStepRecord]:
        step_records = []

        # Reset during the first rollout only
        observations = self.last_observations if self.last_observations else self.env.reset()

        for _ in range(n_steps):
            record = SpacesStepRecord(observations={}, actions={}, rewards={}, dones={}, infos={},
                                      batch_shape=[self.env.n_envs])

            for step_key in self.step_keys:
                record.observations[step_key] = observations

                # Sample action and take the step
                actions = policy.compute_action(observations, policy_id=step_key, maze_state=None)
                observations, rewards, dones, infos = self.env.step(unstack_numpy_list_dict(actions))

                record.actions[step_key] = actions
                record.rewards[step_key] = rewards
                record.dones[step_key] = dones
                record.infos[step_key] = infos

            step_records.append(record)

        return step_records
