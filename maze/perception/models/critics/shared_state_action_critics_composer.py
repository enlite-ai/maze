"""Composer implementation for shared state-action critic."""
from typing import Dict, Union

from gym import spaces
from torch import nn

from maze.core.agent.torch_state_action_critic import TorchSharedStateActionCritic
from maze.core.annotations import override
from maze.core.utils.factory import Factory, CollectionOfConfigType
from maze.core.utils.structured_env_utils import flat_structured_shapes, flat_structured_space
from maze.perception.models.critics.base_state_action_critic_composer import BaseStateActionCriticComposer


class SharedStateActionCriticComposer(BaseStateActionCriticComposer):
    """One critic is shared across all sub-steps or actors (default to use for standard gym-style environments).

    Instantiates a
    :class:`~maze.core.agent.torch_state_action_critic.TorchSharedStateActionCritic`.

    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param action_spaces_dict: Dict of sub-step id to action space.
    :param networks: Critics networks as defined in the config (either list or dictionary of object params and type).
    """

    def __init__(self, observation_spaces_dict: Dict[Union[str, int], spaces.Dict],
                 action_spaces_dict: Dict[Union[str, int], spaces.Dict],
                 networks: CollectionOfConfigType):
        super().__init__(observation_spaces_dict, action_spaces_dict)
        assert len(networks) == 1
        network = networks[0]

        flat_action_space = flat_structured_space(self._action_spaces_dict)
        obs_shapes_flat = flat_structured_shapes(self._obs_shapes)

        # Infer the critic out shapes. When all action heads in a given state are discrete the discrete version of the
        #   state-action critic is used that outputs a value for each possible action (for each action). Otherwise
        #   the more general version is used which returns one value for a given state and action.
        critic_output_shapes = dict()
        if all(self._only_discrete_spaces.values()):
            for act_key, act_space in flat_action_space.spaces.items():
                critic_output_shapes[act_key + '_q_values'] = (act_space.n,)
        else:
            for act_key, act_space in flat_action_space.spaces.items():
                if isinstance(act_space, spaces.Discrete):
                    obs_shapes_flat[act_key] = (act_space.n,)
                else:
                    obs_shapes_flat[act_key] = act_space.sample().shape
            critic_output_shapes['q_value'] = (1,)

        # initialize critic
        model_registry = Factory(base_type=nn.Module)
        self._critics = {0: model_registry.instantiate(network, obs_shapes=obs_shapes_flat,
                                                       output_shapes=critic_output_shapes)}

    @property
    @override(BaseStateActionCriticComposer)
    def critic(self) -> TorchSharedStateActionCritic:
        """implementation of
        :class:`~maze.perception.models.critics.base_state_action_critic_composer.BaseStateActionCriticComposer`
        """
        return TorchSharedStateActionCritic(
            self._critics, num_policies=len(self._obs_shapes), device="cpu",
            only_discrete_spaces={0: all(self._only_discrete_spaces.values())},
            action_spaces_dict={0: flat_structured_space(self._action_spaces_dict)})
