"""Composer implementation for step state-action critic."""

from typing import Dict, Union

from gym import spaces
from torch import nn

from maze.core.agent.torch_state_action_critic import TorchStepStateActionCritic
from maze.core.annotations import override
from maze.core.utils.config_utils import list_to_dict
from maze.core.utils.factory import CollectionOfConfigType, Factory
from maze.perception.models.critics.base_state_action_critic_composer import BaseStateActionCriticComposer


class StepStateActionCriticComposer(BaseStateActionCriticComposer):
    """Each sub-step or actor gets its individual critic.

    Instantiates a
    :class:`~maze.core.agent.torch_state_action_critic.TorchStepStateActionCritic`.

    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param action_spaces_dict: Dict of sub-step id to action space.
    :param networks: Critics networks as defined in the config (either list or dictionary of object params and type).
    """

    def __init__(self, observation_spaces_dict: Dict[Union[str, int], spaces.Dict],
                 action_spaces_dict: Dict[Union[str, int], spaces.Dict],
                 networks: CollectionOfConfigType):
        super().__init__(observation_spaces_dict, action_spaces_dict)

        # Infer the critic out shapes. When all action heads in a given state are discrete the discrete version of the
        #   state-action critic is used that outputs a value for each possible action (for each action). Otherwise
        #   the more general version is used which returns one value for a given state and action.
        critic_output_shapes = dict()
        for step_key, dict_action_space in self._action_spaces_dict.items():
            critic_output_shapes[step_key] = dict()
            if not self._only_discrete_spaces[step_key]:
                for act_key, act_space in dict_action_space.spaces.items():
                    if isinstance(act_space, spaces.Discrete):
                        self._obs_shapes[step_key][act_key] = (act_space.n,)
                    else:
                        self._obs_shapes[step_key][act_key] = act_space.sample().shape
                critic_output_shapes[step_key]['q_value'] = (1,)
            else:
                for act_key, act_space in dict_action_space.spaces.items():
                    critic_output_shapes[step_key][act_key + '_q_values'] = (act_space.n,)

        # initialize critics
        model_registry = Factory(base_type=nn.Module)
        networks = list_to_dict(networks)
        self._critics = {key: model_registry.instantiate(networks[key],
                                                         obs_shapes=self._obs_shapes[key],
                                                         output_shapes=critic_output_shapes[key])
                         for key in networks.keys()}

    @property
    @override(BaseStateActionCriticComposer)
    def critic(self) -> TorchStepStateActionCritic:
        """implementation of
        :class:`~maze.perception.models.critics.base_state_action_critic_composer.BaseStateActionCriticComposer`
        """
        return TorchStepStateActionCritic(self._critics, num_policies=len(self._obs_shapes), device="cpu",
                                          only_discrete_spaces=self._only_discrete_spaces,
                                          action_spaces_dict=self._action_spaces_dict)
