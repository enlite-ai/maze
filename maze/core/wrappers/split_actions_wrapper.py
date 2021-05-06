"""Wrapper for splitting up individual actions."""
from typing import Union, Dict, Tuple, List

import gym
import numpy as np

from maze.core.annotations import override
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import ActionWrapper, EnvType


class SplitActionsWrapper(ActionWrapper[Union[EnvType, StructuredEnvSpacesMixin, StructuredEnv]]):
    """Splits an actions into separate ones.

    An example is given by the LunarLanderContinuous-v2 env. Here we have a box action spaces with shape (2,) such that
    dimension 0 is the up/down action and dimension 1 is the left/right action. Now if we would like to split this
    action correspondingly we can wrap the env with the following config:

    split_config:
        action:
            action_up:
                indices: [0]
            action_side:
                indices: [1]

    Now the actions as well as the action space is consists of two actions (action_up/action_side).

    :param env: Environment/wrapper to wrap.
    :param split_config: The action splitting configuration.
    """

    def __init__(self, env: Union[StructuredEnvSpacesMixin, StructuredEnv],
                 split_config: Dict[str, Dict[str, Dict[str, Union[Tuple, List]]]]):
        super().__init__(env)
        assert isinstance(self.env.action_space, gym.spaces.Dict)
        self._test_config(split_config)
        self.split_config = split_config
        self.reverse_action_ref = {sub_action_name: org_action_name
                                   for org_action_name, sub_action_names in self.split_config.items()
                                   for sub_action_name in sub_action_names}

        # Initialize the action space dict
        self._action_spaces_dict = {step_key: self._split_action_space(action_space)
                                    for step_key, action_space in self.env.action_spaces_dict.items()}

    def _test_config(self, split_config: Dict[str, Dict[str, Dict[str, Union[Tuple, List]]]]) -> None:
        """Test the config for inconsistencies or incomplete definitions.

        :param split_config: The action splitting configuration.
        """
        for action_name, acton_config in split_config.items():
            for new_action_name, new_action_config in acton_config.items():
                split_config[action_name][new_action_name]['indices'] = list(new_action_config['indices'])

        for key in split_config.keys():
            assert key in self.env.action_space.spaces.keys()

        for old_action_name, action_split_config in split_config.items():
            if isinstance(self.env.action_space.spaces[old_action_name], (gym.spaces.Box, gym.spaces.MultiDiscrete,
                                                                          gym.spaces.MultiBinary)):
                org_action_space = self.env.action_space.spaces[old_action_name]
                all_indices = [item for sub_action_config in action_split_config.values()
                               for item in sub_action_config['indices']]
                assert org_action_space.shape[-1] == len(set(all_indices))
            else:
                raise NotImplementedError('The split action wrapper is currently only implemented for box, '
                                          'multi-discrete and multi-binary spaces.')

    def _split_action_space(self, action_space: gym.spaces.Dict) -> gym.spaces.Dict:
        """Split a single action space (dict) into the corresponding sub actions spaces w.r.t. the split config

        :param action_space: The action space to split.
        :return: The resulting (split) action space.
        """
        new_action_space = dict()
        for org_action_name, org_action_space in action_space.spaces.items():
            if org_action_name in self.split_config:
                for new_action_key, new_space_config in self.split_config[org_action_name].items():
                    if isinstance(org_action_space, gym.spaces.Box):
                        low = [org_action_space.low[select_index] for select_index in new_space_config['indices']]
                        high = [org_action_space.high[select_index] for select_index in new_space_config['indices']]
                        new_action_space[new_action_key] = gym.spaces.Box(low=np.asarray(low),
                                                                          high=np.asarray(high),
                                                                          dtype=org_action_space.dtype)
                    elif isinstance(org_action_space, gym.spaces.MultiDiscrete):
                        nvec = [org_action_space.nvec[select_index] for select_index in new_space_config['indices']]
                        if len(nvec) > 1:
                            new_action_space[new_action_key] = gym.spaces.MultiDiscrete(nvec=nvec)
                        else:
                            new_action_space[new_action_key] = gym.spaces.Discrete(n=nvec[0])
                    else:
                        assert isinstance(org_action_space, gym.spaces.MultiBinary)
                        new_action_space[new_action_key] = gym.spaces.MultiBinary(n=len(new_space_config['indices']))
            else:
                new_action_space[org_action_name] = org_action_space
        return gym.spaces.Dict(new_action_space)

    @property
    @override(StructuredEnvSpacesMixin)
    def action_space(self) -> gym.spaces.Dict:
        """The currently active gym action space.
        """
        return self._action_spaces_dict[self.env.actor_id()[0]]

    @property
    @override(StructuredEnvSpacesMixin)
    def action_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Dict]:
        """A dictionary of gym action spaces, with policy IDs as keys.
        """
        return self._action_spaces_dict

    @override(ActionWrapper)
    def action(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ActionWrapper` interface.
        """
        reversed_action = dict()
        tmp_corresponding_actions = dict()
        for action_name, action_value in action.items():
            if action_name in self.reverse_action_ref:
                action_indices = self.split_config[self.reverse_action_ref[action_name]][action_name]['indices']
                if self.reverse_action_ref[action_name] not in tmp_corresponding_actions:
                    tmp_corresponding_actions[self.reverse_action_ref[action_name]] = dict()
                for elem, index in enumerate(action_indices):
                    tmp_corresponding_actions[self.reverse_action_ref[action_name]][index] = action_value[elem]
            else:
                reversed_action[action_name] = action_value

        for org_action_name, split_action in tmp_corresponding_actions.items():
            org_action = np.stack([split_action[index] for index in range(len(split_action))])
            reversed_action[org_action_name] = org_action

        return reversed_action

    @override(ActionWrapper)
    def reverse_action(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ActionWrapper` interface.
        """
        new_action = {}
        for org_action_name, org_action in action.items():
            if org_action_name in self.split_config:
                for new_action_key, new_action_config in self.split_config[org_action_name].items():
                    new_action[new_action_key] = np.stack([org_action[select_index] for select_index in
                                                           new_action_config['indices']])
            else:
                new_action[org_action_name] = action[org_action_name]

        return new_action

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'SplitActionsWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
