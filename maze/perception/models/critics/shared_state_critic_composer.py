"""Composer implementation for shared critic."""
from typing import Dict

from gym import spaces
from torch import nn

from maze.core.agent.torch_state_critic import TorchSharedStateCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StepKeyType
from maze.core.utils.factory import Factory, ConfigType
from maze.core.utils.structured_env_utils import flat_structured_shapes, stacked_shapes
from maze.perception.models.critics.base_state_critic_composer import BaseStateCriticComposer


class SharedStateCriticComposer(BaseStateCriticComposer):
    """One critic is shared across all sub-steps or actors (default to use for standard gym-style environments).

    Instantiates a
    :class:`~maze.core.agent.torch_state_critic.TorchSharedStateCritic`.

    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param networks: The single, shared critic network as defined in the config.
    """

    def __init__(self,
                 observation_spaces_dict: Dict[StepKeyType, spaces.Dict],
                 agent_counts_dict: Dict[StepKeyType, int],
                 networks: ConfigType,
                 stack_observations: bool):
        super().__init__(observation_spaces_dict, agent_counts_dict)
        assert len(networks) == 1
        self.stack_observations = stack_observations
        network = networks[0]

        obs_shapes_flat = self._obs_shapes
        if self.stack_observations:
            obs_shapes_flat = stacked_shapes(obs_shapes_flat, self._agent_counts_dict)
        obs_shapes_flat = flat_structured_shapes(obs_shapes_flat)

        # initialize critic
        model_registry = Factory(base_type=nn.Module)
        self._critics = {0: model_registry.instantiate(network, obs_shapes=obs_shapes_flat)}

    @property
    @override(BaseStateCriticComposer)
    def critic(self) -> TorchSharedStateCritic:
        """implementation of :class:`~maze.perception.models.critics.base_state_critic_composer.BaseStateCriticComposer`
        """
        return TorchSharedStateCritic(self._critics, num_policies=len(self._obs_shapes), device="cpu",
                                      stack_observations=self.stack_observations)
