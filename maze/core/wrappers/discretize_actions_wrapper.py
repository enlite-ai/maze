"""Wrapper for discretizing individual actions."""
from typing import Union, Dict, List

import gym
import numpy as np

from maze.core.annotations import override
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.utils.structured_env_utils import flat_structured_space
from maze.core.wrappers.wrapper import ActionWrapper, EnvType


class DiscretizeActionsWrapper(ActionWrapper[Union[EnvType, StructuredEnvSpacesMixin, StructuredEnv]]):
    """The DiscretizeActionsWrapper provides functionality for discretizing individual continuous actions into discrete
     ones.

    An example is given by having a continuous action called 'action_up' with space:
    gym.spaces.Box(shape=(5,), low=[-1,-1,-1,-1,-1], high=[1,1,1,1,1]

    discretization_config:
        action_up:
            num_bins: 5
            low: [-1, 0, 0.5, 0, 0]
            high: 1

    Now the action space will be split where each of the 5 continuous values of the box spaces are split evenly within
    the ranges of (-1,1),(0,1),(0.5,1), (0,1), (0,1) respectively.

    :param env: Environment/wrapper to wrap.
    :param discretization_config: The action discretization configuration.
    """

    def __init__(self, env: Union[StructuredEnvSpacesMixin, StructuredEnv],
                 discretization_config: Dict[str, Dict[str, Union[Union[int, float], List[Union[int, float]]]]]):
        super().__init__(env)
        assert isinstance(self.env.action_space, gym.spaces.Dict)
        self._test_config(discretization_config)
        self.discretization_config = discretization_config

        self._bins: Dict[str, np.ndarray] = dict()
        self._avgs: Dict[str, np.ndarray] = dict()

        # Initialize the action space dict
        self._action_spaces_dict = {step_key: self._discretize_action_space(action_space)
                                    for step_key, action_space in self.env.action_spaces_dict.items()}

    def _test_config(self, discretization_config: Dict[str, Dict[str, Union[Union[int, float],
                                                                            List[Union[int, float]]]]]) -> None:
        """Test the config for inconsistencies or incomplete definitions.

        :param discretization_config: The action discretization configuration.
        """

        for action_key, config in discretization_config.items():
            # Assert key is present in action space
            flat_action_space = flat_structured_space(self.env.action_spaces_dict)
            assert action_key in flat_action_space.spaces.keys(), f'{action_key} not in {flat_action_space}'
            assert isinstance(flat_action_space[action_key], gym.spaces.Box)
            assert len(flat_action_space[action_key].shape) == 1, \
                'Only single dimensional continuous spaces supported. Please use the SplitActionWrapper in order to ' \
                'split the actions first.'
            assert 'num_bins' in config
            assert config['num_bins'] > 1, 'Num of bins must be greater than 1'
            if 'low' in config:
                assert 'high' in config
                if isinstance(config['low'], (float, int)) and not isinstance(config['high'], (float, int)):
                    high = np.array(config['high'])
                    low = np.array([config['low']] * len(high))
                elif isinstance(config['high'], (float, int)) and not isinstance(config['low'], (float, int)):
                    low = np.array(config['low'])
                    high = np.array([config['high']] * len(low))
                else:
                    low = config['low']
                    high = config['high']
                    if not isinstance(low, (float, int)):
                        assert len(low) == len(high)
                assert np.all(low < high), f'lower bound must be smaller than higher bound (low: {low}, high: {high}' \
                                           f', action_key: {action_key})'
                assert np.all(low >= flat_action_space[action_key].low), \
                    f'Lower bound has to be larger or equal to the lower bound of the original space (low: {low}, ' \
                    f'original space low: {flat_action_space[action_key].low}, action_key: {action_key})'
                assert np.all(high <= flat_action_space[action_key].high), \
                    f'Higher bound has to be smaller or equal to the higher bound of the original space (high: {high}' \
                    f', original space high: {flat_action_space[action_key].high}, action_key: {action_key})'

    def _discretize_action_space(self, action_space: gym.spaces.Dict) -> gym.spaces.Dict:
        """Discretize a single action space (dict) into the corresponding sub actions spaces w.r.t. the split config

        :param action_space: The action space to split.
        :return: The resulting (split) action space.
        """
        new_action_space = dict()
        for org_action_name, org_action_space in action_space.spaces.items():
            if org_action_name in self.discretization_config:
                dis_config = self.discretization_config[org_action_name]
                if 'low' in dis_config:
                    low = dis_config['low']
                    if isinstance(dis_config['low'], (float, int)):
                        low = np.array([dis_config['low']] * org_action_space.low.shape[-1])
                    high = dis_config['high']
                    if isinstance(dis_config['high'], (float, int)):
                        high = np.array([dis_config['high']] * org_action_space.high.shape[-1])
                else:
                    low = org_action_space.low
                    high = org_action_space.high
                bounds = np.linspace(start=low, stop=high, num=dis_config['num_bins'] + 1)
                self._bins[org_action_name] = bounds[1:]
                self._avgs[org_action_name] = np.array(list(zip(bounds, bounds[1:])), dtype=np.float32).mean(axis=1)
                assert self._bins[org_action_name].shape == self._avgs[org_action_name].shape
                if org_action_space.shape[-1] == 1:
                    new_action_space[org_action_name] = gym.spaces.Discrete(n=dis_config['num_bins'])
                else:
                    nvec = np.array([dis_config['num_bins']] * org_action_space.shape[-1])
                    new_action_space[org_action_name] = gym.spaces.MultiDiscrete(nvec=nvec)
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
        new_action = {}
        for org_action_name, org_action in action.items():
            if org_action_name in self.discretization_config and isinstance(org_action, np.ndarray) \
                    and len(org_action.shape) > 0:
                new_action[org_action_name] = self._avgs[org_action_name][org_action, range(len(org_action))]
            elif org_action_name in self.discretization_config:
                new_action[org_action_name] = self._avgs[org_action_name][org_action]
            else:
                new_action[org_action_name] = org_action

        return new_action

    @override(ActionWrapper)
    def reverse_action(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ActionWrapper` interface.
        """
        my_func = np.vectorize(np.digitize, signature='(),(m),()->()')
        new_action = {}
        for org_action_name, org_action in action.items():
            if org_action_name in self.discretization_config:
                out = my_func(org_action, self._bins[org_action_name].T, right=True)
                if len(out) == 1:
                    out = out[0]
                new_action[org_action_name] = out
            else:
                new_action[org_action_name] = org_action

        return new_action

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'DiscretizeActionsWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
