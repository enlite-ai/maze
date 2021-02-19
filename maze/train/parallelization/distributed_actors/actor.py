"""Actor agent common for all distributed actors modules. Encapsulates a policy and steps through the environment
given, recording the rollouts."""

import collections
from typing import Callable, Dict, NoReturn

import numpy as np
import torch

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.structured_env import StructuredEnv
from maze.core.log_stats.log_stats import LogStatsLevel
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.perception.perception_utils import convert_to_torch
from maze.train.utils.train_utils import stack_torch_dict_list

ActorOutput = collections.namedtuple('AgentOutput', 'observations rewards dones infos actions_taken actions_logits')
"""Datatype for agent outputs."""
ActorOutput_w_stats = collections.namedtuple('AgentOutput_w_stats', 'observations rewards dones infos actions_taken '
                                                                    'actions_logits stats')
"""Datatype for agent outputs with statistics."""


class ActorAgent:
    """Steps through a given environment and records rollouts. Designed to be used in distributed rollouts."""

    def __init__(self,
                 env_factory: Callable,
                 policy: TorchPolicy,
                 n_rollout_steps: int):
        self.env: StructuredEnv = env_factory()
        self.policy = policy
        self.n_rollout_steps = n_rollout_steps

        if not isinstance(self.env, LogStatsWrapper):
            self.env = LogStatsWrapper.wrap(self.env)

        self.last_obs = self.env.reset()

        self.sub_step_keys = list(self.policy.networks.keys())

    def update_policy(self, state_dict: Dict) -> NoReturn:
        """Update the policy with the given state dict.

        :param state_dict: State dict to load.
        """
        self.policy.load_state_dict(state_dict)

    def rollout(self) -> ActorOutput_w_stats:
        """Performs a agent rollout, that is sample actions and step through the env for a maximum of
        n_rollout_steps. This rollout (observations, rewards, dones, infos, actions_taken, actions_logits) is returned

        :return: This rollout (observations, rewards, dones, infos, actions_taken, actions_logits) as an ActorOutput
            named tuple
        """
        # Perform agent rollout without collecting gradients

        # shared lists
        rewards, dones, infos = [], [], []

        # step specific lists
        step_observations = {key: [] for key in self.sub_step_keys}
        step_actions_taken = {key: [] for key in self.sub_step_keys}
        step_action_logits = {key: [] for key in self.sub_step_keys}
        stats = []

        for i in range(self.n_rollout_steps):
            # set initial observation
            obs = self.last_obs

            # sample step actions
            assert len(self.sub_step_keys) > 0
            reward, done, info = None, None, None
            for _ in range(len(self.sub_step_keys)):
                step_key, _ = self.env.actor_id()
                step_observations[step_key].append(convert_to_torch(
                    obs, in_place='try', device=self.policy.device, cast=None))
                action, logits = self.policy.compute_action_with_logits(obs, step_key, deterministic=False)
                obs, reward, done, info = self.env.step(action)
                stats.append(self.env.get_stats(LogStatsLevel.EPISODE).last_stats)

                if done:
                    info['terminal_observation'] = obs
                    obs = self.env.reset()

                # keep action taken
                step_actions_taken[step_key].append(convert_to_torch(
                    action, in_place='try', device=self.policy.device, cast=None))
                step_action_logits[step_key].append(logits)

            # update previous observation
            self.last_obs = obs

            # book keeping
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        observations = {step_key: stack_torch_dict_list(o, expand=True) for step_key, o in step_observations.items()}
        rewards = convert_to_torch(np.vstack(rewards), in_place='try', device=self.policy.device, cast=None)
        dones = convert_to_torch(np.vstack(dones), cast=torch.bool, in_place='try', device=self.policy.device)
        actions_taken = {step_key: stack_torch_dict_list(at, expand=True)
                         for step_key, at in step_actions_taken.items()}
        actions_logits = {step_key: stack_torch_dict_list(al, expand=True) for
                          step_key, al in step_action_logits.items()}

        return ActorOutput_w_stats(observations, rewards, dones, infos, actions_taken, actions_logits, stats)
