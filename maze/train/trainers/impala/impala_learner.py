"""Implementation of the Impala Learner"""
import collections
from typing import Dict, Union

import gym
import numpy as np
import torch

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.env.structured_env import StructuredEnv, ActorID
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.train.parallelization.vector_env.vector_env import VectorEnv

LearnerOutput = collections.namedtuple('LearnerOutput', 'values detached_values actions_logits')


class ImpalaLearner:
    """Learner agent for Impala. The agent only exists once (in the main thread) and is in charge of doing the loss
        computation as computing and backpropagating the gradients. Furthermore it holds critic network in contrast to
        the actors.
        """

    def __init__(self,
                 eval_env: Union[VectorEnv, StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv],
                 model: TorchActorCritic,
                 n_rollout_steps: int):
        self.env = eval_env
        self.model = model
        self.n_rollout_steps = n_rollout_steps

        self.sub_step_keys = list(self.model.policy.networks.keys())

        self.step_action_keys = {}
        for step_key, step_space in self.env.action_spaces_dict.items():
            assert isinstance(step_space, gym.spaces.Dict)
            self.step_action_keys[step_key] = list(step_space.spaces.keys())

    def learner_rollout_on_agent_output(self, actors_output: StructuredSpacesRecord) -> LearnerOutput:
        """Compute the values and the action logits using the learners network parameters and the actors rollouts.
            Thus we never step through an env here.

        :param actors_output: The collected and batched actors output, including the env_outputs such as observations
            and actions

        :return: A LearnerOutput names tuple consisting of (values, detached_values, actions_logits, n_critics)
        """

        # predict values of stage 2 in a regular fashion
        values, detached_values = self.model.critic.predict_values(actors_output)

        # compute action log-probabilities of actions taken
        actions_logits = []
        for record in actors_output.substep_records:
            substep_logits = self.model.policy.compute_logits_dict(record.observation, actor_id=record.actor_id)
            actions_logits.append(substep_logits)

        # convert logits and values into dicts keyed by sub-step ID
        values = {r.substep_key: v for r, v in zip(actors_output.substep_records, values)}
        detached_values = {r.substep_key: dv for r, dv in zip(actors_output.substep_records, detached_values)}
        actions_logits = {r.substep_key: al for r, al in zip(actors_output.substep_records, actions_logits)}

        return LearnerOutput(values, detached_values, actions_logits)

    def evaluate(self, deterministic: bool, repeats: int) -> None:
        """Perform evaluation on eval env.

        :param deterministic: deterministic or stochastic action sampling (selection)
        :param repeats: number of evaluation episodes to average over
        """

        dones_count = 0
        dones = []
        obs = self.env.reset()
        while dones_count < repeats:

            # iterate environment steps
            for step_key in self.sub_step_keys:
                sampled_action = self.model.policy.compute_action(obs, actor_id=ActorID(step_key, 0),
                                                                  deterministic=deterministic)
                obs, step_rewards, dones, infos = self.env.step(sampled_action)

            if np.any(dones):
                dones_count += np.count_nonzero(dones)

        # enforce the epoch stats calculation instead of waiting for the next increment_log_step() call
        self.env.write_epoch_stats()
