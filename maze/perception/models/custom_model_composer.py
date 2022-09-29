"""Custom model composer, encapsulating the set of policy and critic networks along with the distribution mapper."""
import copy
from typing import Dict, Union, Optional, Mapping

import gym
import numpy as np
from gym import spaces
from maze.core.agent.state_critic_input_output import StateCriticStepInput
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_action_critic import TorchStateActionCritic
from maze.core.agent.torch_state_critic import TorchStateCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StepKeyType, ActorID
from maze.core.utils.factory import Factory, ConfigType
from maze.perception.models.critics import BaseStateCriticComposer, SharedStateCriticComposer
from maze.perception.models.critics.base_state_action_critic_composer import BaseStateActionCriticComposer
from maze.perception.models.critics.critic_composer_interface import CriticComposerInterface
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.models.policies.base_policy_composer import BasePolicyComposer
from maze.perception.perception_utils import map_nested_structure


class CustomModelComposer(BaseModelComposer):
    """Composes models from explicit model definitions.

    :param action_spaces_dict: Dict of sub-step id to action space.
    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param distribution_mapper_config: Distribution mapper configuration.
    :param policy: Mapping of sub-step keys to models.
    :param critic: Configuration for the critic composer.
    """

    @classmethod
    def check_model_config(cls, model_config: ConfigType) -> None:
        """Asserts the provided model config for consistency.
        :param model_config: The model config to check.
        """
        if 'policy' in model_config and model_config['policy'] and not isinstance(
                model_config["policy"], BasePolicyComposer):
            assert '_target_' in model_config['policy']
            assert 'networks' in model_config['policy'], \
                f"Custom models expect explicit policy networks! Check the model config!"
        if 'critic' in model_config and model_config["critic"] and not isinstance(
                model_config["critic"], CriticComposerInterface
        ):
            assert '_target_' in model_config['critic']
            assert 'networks' in model_config['critic'], \
                f"Custom models expect explicit critic networks! Check the model config!"

    def __init__(self,
                 action_spaces_dict: Dict[StepKeyType, gym.spaces.Dict],
                 observation_spaces_dict: Dict[StepKeyType, gym.spaces.Dict],
                 agent_counts_dict: Dict[StepKeyType, int],
                 distribution_mapper_config: ConfigType,
                 policy: ConfigType,
                 critic: ConfigType):
        super().__init__(action_spaces_dict, observation_spaces_dict, agent_counts_dict, distribution_mapper_config)

        # init policy composer
        self._policy_composer = Factory(BasePolicyComposer).instantiate(
            policy,
            action_spaces_dict=self.action_spaces_dict,
            observation_spaces_dict=self.observation_spaces_dict,
            agent_counts_dict=self.agent_counts_dict,
            distribution_mapper=self._distribution_mapper
        )

        self.critic_input_spaces_dict = self._build_critic_input_space_dict()

        # init critic composer
        self._critics_composer = None
        if critic is not None:
            critic_type = Factory(
                CriticComposerInterface
            ).type_from_name(critic['_target_']) if isinstance(critic, Mapping) else type(critic)
            if issubclass(critic_type, SharedStateCriticComposer):
                assert self.critic_input_spaces_dict == self.observation_spaces_dict, \
                    f'Shared embedding is not yet supported for shared state critics'

            if issubclass(critic_type, BaseStateCriticComposer):
                self._critics_composer = Factory(BaseStateCriticComposer).instantiate(
                    critic,
                    observation_spaces_dict=self.critic_input_spaces_dict,
                    agent_counts_dict=self.agent_counts_dict
                )
            elif issubclass(critic_type, BaseStateActionCriticComposer):
                assert self.critic_input_spaces_dict == self.observation_spaces_dict, \
                    f'Shared embedding is not yet supported for state-action critics'
                self._critics_composer = Factory(BaseStateActionCriticComposer).instantiate(
                    critic, observation_spaces_dict=self.critic_input_spaces_dict,
                    action_spaces_dict=self.action_spaces_dict
                )
            else:
                raise ValueError(f"Critic of type {critic_type} not supported!")

        # save model graphs to pdf
        self.save_models()

    def _build_critic_input_space_dict(self) -> Dict[StepKeyType, spaces.Dict]:
        """Build the critic input from the given observation input and a dummy pass through the policy network (in case
        shared embeddings are used).

        :return: The dict holding the enw critic input spaces dict, needed for building the model.
        """
        critic_input_spaces_dict = dict(copy.deepcopy(self.observation_spaces_dict))
        for step_key, obs_space in self.observation_spaces_dict.items():
            step_observation = dict()
            for obs_key, obs in obs_space.spaces.items():
                if isinstance(obs, spaces.Box) and np.any(obs_space[obs_key].low == np.finfo(np.float32).min) or \
                        np.any(obs_space[obs_key].high == np.finfo(np.float32).max):
                    # In case any of the lower or upper bounds of the space are infinite, resample the values.
                    step_observation[obs_key] = np.random.randn(*obs.shape).astype(np.float32)
                else:
                    # Set random generator to None.. In case the observation spaces have been loaded from
                    #   a file not setting this may lead to problems
                    obs._np_random = None
                    step_observation[obs_key] = obs.sample()
            if self._policy_composer is not None:
                tmp_out = self._policy_composer.policy.compute_substep_policy_output(
                    step_observation, actor_id=ActorID(step_key, 0))
                if tmp_out.embedding_logits is not None:
                    new_observation_space = dict()
                    critic_input = StateCriticStepInput.build(tmp_out, step_observation)
                    for in_key, in_value in critic_input.tensor_dict.items():
                        if in_key in critic_input_spaces_dict[step_key]:
                            new_observation_space[in_key] = critic_input_spaces_dict[step_key][in_key]
                        else:
                            new_observation_space[in_key] = spaces.Box(low=np.finfo(np.float32).min,
                                                                       high=np.finfo(np.float32).max,
                                                                       shape=in_value.shape, dtype=np.float32)
                    critic_input_spaces_dict[step_key] = gym.spaces.Dict(dict(new_observation_space))
        return critic_input_spaces_dict

    @property
    @override(BaseModelComposer)
    def policy(self) -> Optional[TorchPolicy]:
        """Return the policy networks."""
        if self._policy_composer is None:
            return None

        return self._policy_composer.policy

    @property
    @override(BaseModelComposer)
    def critic(self) -> Optional[Union[TorchStateCritic, TorchStateActionCritic]]:
        """Return the critic networks."""
        if self._critics_composer is None:
            return None

        return self._critics_composer.critic
