"""Custom model composer, encapsulating the set of policy and critic networks along with the distribution mapper."""

from typing import Dict, Union, Optional, Mapping, List

import gym
import numpy as np
from gym import spaces

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_action_critic import TorchStateActionCritic
from maze.core.agent.torch_state_critic import TorchStateCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StepKeyType
from maze.core.utils.factory import Factory, ConfigType
from maze.perception.models.critics import BaseStateCriticComposer
from maze.perception.models.critics.base_state_action_critic_composer import BaseStateActionCriticComposer
from maze.perception.models.critics.critic_composer_interface import CriticComposerInterface
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.models.policies.base_policy_composer import BasePolicyComposer


class CustomModelComposer(BaseModelComposer):
    """Composes models from explicit model definitions.

    :param action_spaces_dict: Dict of sub-step id to action space.
    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param distribution_mapper_config: Distribution mapper configuration.
    :param policy: Mapping of sub-step keys to models.
    :param critic: Configuration for the critic composer.
    :param shared_embedding_keys: Specify up to which keys to share the embedding between actor and critic
        (if applicable).
    """

    @classmethod
    def check_model_config(cls, model_config: ConfigType) -> None:
        """Asserts the provided model config for consistency.
        :param model_config: The model config to check.
        """
        if 'policy' in model_config and not isinstance(model_config["policy"], BasePolicyComposer):
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
                 critic: ConfigType,
                 shared_embedding_keys: Optional[List[str]]):
        super().__init__(action_spaces_dict, observation_spaces_dict, agent_counts_dict, distribution_mapper_config)

        self.shared_embedding = shared_embedding_keys is not None and shared_embedding_keys is not []

        # init policy composer
        self._policy_composer = Factory(BasePolicyComposer).instantiate(
            policy,
            action_spaces_dict=self.action_spaces_dict,
            observation_spaces_dict=self.observation_spaces_dict,
            agent_counts_dict=self.agent_counts_dict,
            distribution_mapper=self._distribution_mapper
        )

        embedding_out = dict()
        if self.shared_embedding:
            for step_key, observation_space in observation_spaces_dict.items():
                actor_id = (step_key, 0) if self._policy_composer.policy.substeps_with_separate_agent_nets else step_key
                out, embedding_out[step_key] = self._policy_composer.policy.compute_logits_dict(
                    observation_space.sample(), actor_id=actor_id, return_embedding=self.shared_embedding)

        # init critic composer
        self._critics_composer = None
        if critic is not None:
            critic_type = Factory(
                CriticComposerInterface
            ).type_from_name(critic['_target_']) if isinstance(critic, Mapping) else type(critic)

            if issubclass(critic_type, BaseStateCriticComposer):
                if self.shared_embedding:
                    observation_spaces_dict = {step_key: spaces.Dict(
                        {latent_key: spaces.Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max,
                                                shape=latent_tensor.shape, dtype=np.float32)
                         for latent_key, latent_tensor in embedding_step_out.items()})
                        for step_key, embedding_step_out in embedding_out.items()}
                self._critics_composer = Factory(BaseStateCriticComposer).instantiate(
                    critic,
                    observation_spaces_dict=self.observation_spaces_dict,
                    agent_counts_dict=self.agent_counts_dict,
                    shared_embedding=self.shared_embedding
                )
            elif issubclass(critic_type, BaseStateActionCriticComposer):
                assert not self.shared_embedding, 'Shared embedding is not supported for state action critics'
                self._critics_composer = Factory(BaseStateActionCriticComposer).instantiate(
                    critic, observation_spaces_dict=self.observation_spaces_dict,
                    action_spaces_dict=self.action_spaces_dict
                )
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
    def critic(self) -> Optional[Union[TorchStateCritic, TorchStateActionCritic]]:
        """Return the critic networks."""
        if self._critics_composer is None:
            return None

        return self._critics_composer.critic
