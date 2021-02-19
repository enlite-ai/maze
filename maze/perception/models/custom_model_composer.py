"""Custom model composer, encapsulating the set of policy and critic networks along with the distribution mapper."""

from typing import Dict, Union, Optional

import gym

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_critic import TorchStateCritic
from maze.core.annotations import override
from maze.core.utils.registry import Registry, ConfigType
from maze.perception.models import critics as critics_module
from maze.perception.models import policies as policies_module
from maze.perception.models.critics import BaseStateCriticComposer
from maze.perception.models.critics.base_state_action_critic_composer import BaseStateActionCriticComposer
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.models.policies.base_policy_composer import BasePolicyComposer


class CustomModelComposer(BaseModelComposer):
    """Composes models from explicit model definitions.

    :param action_spaces_dict: Dict of sub-step id to action space.
    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param distribution_mapper_config: Distribution mapper configuration.
    :param policy: Mapping of sub-step keys to models.
    :param critic: Configuration for the critic composer.
    """

    critics_registry = Registry(base_type=BaseStateCriticComposer, root_module=critics_module)
    policy_registry = Registry(base_type=BasePolicyComposer, root_module=policies_module)

    @classmethod
    def check_model_config(cls, model_config: ConfigType) -> None:
        """Asserts the provided model config for consistency.
        :param model_config: The model config to check.
        """
        if 'policy' in model_config:
            assert 'type' in model_config['policy']
            assert 'networks' in model_config['policy'], \
                f"Custom models expect explicit policy networks! Check the model config!"
        if 'critic' in model_config and model_config["critic"]:
            assert 'type' in model_config['critic']
            assert 'networks' in model_config['critic'], \
                f"Custom models expect explicit critic networks! Check the model config!"

    def __init__(self,
                 action_spaces_dict: Dict[Union[str, int], gym.spaces.Dict],
                 observation_spaces_dict: Dict[Union[str, int], gym.spaces.Dict],
                 distribution_mapper_config: ConfigType,
                 policy: ConfigType,
                 critic: ConfigType):
        super().__init__(action_spaces_dict, observation_spaces_dict, distribution_mapper_config)

        # init policy composer
        self._policy_composer = self.policy_registry.arg_to_obj(policy,
                                                                action_spaces_dict=self.action_spaces_dict,
                                                                observation_spaces_dict=self.observation_spaces_dict,
                                                                distribution_mapper=self._distribution_mapper)

        # init critic composer
        self._critics_composer = None
        critic_type = self.critics_registry.class_type_from_module_name(critic['type']) \
            if critic is not None else None
        if critic_type:
            if issubclass(critic_type, BaseStateCriticComposer):
                self._critics_composer = \
                    self.critics_registry.arg_to_obj(critic, observation_spaces_dict=self.observation_spaces_dict)
            elif issubclass(critic_type, BaseStateActionCriticComposer):
                self._critics_composer = \
                    self.critics_registry.arg_to_obj(critic, observation_spaces_dict=self.observation_spaces_dict,
                                                     action_spaces_dict=self.action_spaces_dict)
            else:
                raise ValueError(f"Critic of type {critic_type} not supported!")

        # save model graphs to pdf
        self.save_models()

    @property
    @override(BaseModelComposer)
    def policy(self) -> Optional[TorchPolicy]:
        """Return the policy networks."""
        if self._policy_composer is None:
            return None

        return self._policy_composer.policy

    @property
    @override(BaseModelComposer)
    def critic(self) -> Optional[TorchStateCritic]:
        """Return the critic networks."""
        if self._critics_composer is None:
            return None

        return self._critics_composer.critic
