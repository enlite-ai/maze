"""Contains a dictionary action space wrapper."""
from typing import Dict, Tuple, Union

import gym
import numpy as np
from maze.core.annotations import override
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin

from maze.core.wrappers.wrapper import ActionWrapper


class DictActionWrapper(ActionWrapper[gym.Env]):
    """Wraps either a single action space or a tuple action space into dictionary space.

    :param env: The environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        assert not isinstance(env.action_space, gym.spaces.Dict), "Action spaces is already a dict space!"

        # remember original action space
        self._original_space = env.action_space

        # tuple action spaces
        if isinstance(self._original_space, gym.spaces.Tuple):
            self._space_dict = dict()
            for i, space in enumerate(self._original_space.spaces):
                self._space_dict[f"action_{i}"] = space
            self.action_space = gym.spaces.Dict(self._space_dict)
        # single action spaces
        else:
            self._space_dict = {"action": self._original_space}
            self.action_space = gym.spaces.Dict(self._space_dict)

    def action(self, action: Dict[str, np.ndarray]) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ActionWrapper` interface.
        """
        if isinstance(self._original_space, gym.spaces.Tuple):
            return tuple([v for v in action.values()])
        else:
            return action["action"]

    def reverse_action(self, action: Union[np.ndarray, Tuple[np.ndarray]]) -> Dict[str, np.ndarray]:
        """Implementation of :class:`~maze.core.wrappers.wrapper.ActionWrapper` interface.
        """
        if isinstance(self._original_space, gym.spaces.Tuple):
            dict_action = dict()
            for i, action in enumerate(action):
                dict_action[f"action_{i}"] = action
            return dict_action
        else:
            return {"action": action}

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'DictActionWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
