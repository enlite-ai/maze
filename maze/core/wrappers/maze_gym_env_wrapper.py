"""Contains wrapper transforming a standard gym environment into a maze environment."""
from abc import ABC
from copy import copy, deepcopy
from typing import Tuple, Union, Any, Dict, Optional, List, Type

import gym
import numpy as np
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv, Continuous_MountainCarEnv, PendulumEnv, AcrobotEnv
from gym.wrappers import TimeLimit

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionConversionInterface
from maze.core.env.core_env import CoreEnv
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.reward import RewardAggregatorInterface
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationConversionInterface
from maze.core.env.structured_env import StepKeyType, ActorID
from maze.core.events.pubsub import Subscriber
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.rendering.renderer import Renderer

try:
    from gym.envs.atari import AtariEnv
except gym.error.DependencyNotInstalled:
    AtariEnv = None


class GymActionConversion(ActionConversionInterface):
    """A dummy conversion interface asserting that the action is packed into a dictionary space.

    :param env: Gym environment.
    """

    def __init__(self, env: gym.Env):
        self.env = env
        # We need to know in the method `space_to_maze` if the action space is a Dict action space.
        self._original_space_is_dict = isinstance(env.action_space, gym.spaces.Dict)

    @override(ActionConversionInterface)
    def space_to_maze(self, action: Dict[str, np.ndarray], maze_state: MazeStateType) -> MazeActionType:
        """Converts agent action to environment MazeAction.

        :param action: the agent action.
        :param maze_state: the environment state.
        :return: the environment MazeAction.
        """
        if not self._original_space_is_dict:
            maze_action = action["action"]
        else:
            maze_action = action

        return maze_action

    @override(ActionConversionInterface)
    def maze_to_space(self, maze_action: MazeActionType) -> Dict[str, np.ndarray]:
        """Converts environment MazeAction to agent action.

        :param maze_action: the environment MazeAction.
        :return: the agent action.
        """
        if not self._original_space_is_dict:
            maze_action = {"action": maze_action}
        return maze_action

    @override(ActionConversionInterface)
    def space(self) -> gym.spaces.Dict:
        """Returns respective gym action space.
        """
        if not self._original_space_is_dict:
            action_space = gym.spaces.Dict({"action": self.env.action_space})
        else:
            action_space = self.env.action_space

        return action_space


class GymObservationConversion(ObservationConversionInterface):
    """A dummy conversion interface asserting that the observation is packed into a dictionary space.

    :param env: Gym environment.
    """

    def __init__(self, env: gym.Env):
        self.env = env
        # We need to know in the method `space_to_maze` if the observation space is a Dict observation space.
        self._original_space_is_dict = isinstance(env.observation_space, gym.spaces.Dict)

    @override(ObservationConversionInterface)
    def maze_to_space(self, maze_state: MazeStateType) -> MazeStateType:
        """Converts core environment state to agent observation.
        """
        if not self._original_space_is_dict:
            maze_state = {"observation": maze_state.astype(np.float32)}
        return maze_state

    @override(ObservationConversionInterface)
    def space_to_maze(self, observation: Dict[str, np.ndarray]) -> MazeStateType:
        """Converts agent observation to core environment state.
        (This is most like not possible for most observation observation_conversion)
        """
        return observation["observation"]

    @override(ObservationConversionInterface)
    def space(self) -> gym.spaces.Space:
        """Returns respective gym observation space.
        """
        if not self._original_space_is_dict:
            observation_space = gym.spaces.Dict({"observation": self.env.observation_space})
        else:
            observation_space = self.env.observation_space

        return observation_space


class GymRewardAggregator(RewardAggregatorInterface):
    """A dummy reward aggregation object simply repeating the environment's original reward.
    """

    @override(Subscriber)
    def get_interfaces(self) -> List[Type[ABC]]:
        """Nothing to do here
        """

    @classmethod
    @override(RewardAggregatorInterface)
    def to_scalar_reward(cls, reward: float) -> float:
        """Nothing to do here for this env.

        :param: reward: already a scalar reward
        :return: the same scalar reward
        """
        return reward


class GymRenderer(Renderer):
    """A Maze-style Gym renderer.

    Note: Not yet compatible with Maze offline rendering tools (i.e., while gym envs can be rendered during a rollout,
    they do not support offline rendering, such as in the Trajectory Viewer notebook).
    """

    def __init__(self, env: gym.Env):
        self.env = env

    @override(Renderer)
    def render(self, maze_state: MazeStateType, maze_action: Optional[MazeActionType], events: StepEventLog, **kwargs) -> None:
        """Render the current state of the environment."""
        assert self.env is not None, "'GymMazeEnv' renderer is not yet fully compatible with the Maze suite of " \
                                     "rendering tools."
        self.env.render()

    def __getstate__(self) -> dict:
        """Skip env when pickling this class (this renderer is not yet compatible with Maze offline rendering tools)"""
        obj_dict = copy(self.__dict__)
        obj_dict.pop("env", None)
        return obj_dict


class GymCoreEnv(CoreEnv):
    """Wraps a Gym environment into a maze core environment.

    :param env: The Gym environment.
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env

        # initialize renderer
        self.renderer = GymRenderer(env)

        # initialize reward aggregator
        self.reward_aggregator = GymRewardAggregator()

        # initialize the state
        self._maze_state: Optional[Dict] = None

    def step(self, maze_action: MazeActionType) -> Tuple[MazeStateType, Union[float, np.ndarray, Any], bool, Dict[Any, Any]]:
        """Intercept ``CoreEnv.step``"""
        maze_state, rew, done, info = self.env.step(maze_action)
        self._maze_state = maze_state
        return maze_state, rew, done, info

    @override(CoreEnv)
    def get_renderer(self) -> Renderer:
        """Intercept ``CoreEnv.get_renderer``"""
        return self.renderer

    @override(CoreEnv)
    def get_serializable_components(self) -> Dict[str, Any]:
        """Intercept ``CoreEnv.get_serializable_components``"""
        return {}

    @override(CoreEnv)
    def get_maze_state(self) -> MazeStateType:
        """Intercept ``CoreEnv.get_maze_state``"""
        return self._maze_state

    @override(CoreEnv)
    def close(self) -> None:
        """Intercept ``CoreEnv.close``"""
        self.env.close()

    @override(CoreEnv)
    def reset(self) -> MazeStateType:
        """Intercept ``CoreEnv.reset``"""
        maze_state = self.env.reset()
        self._maze_state = maze_state
        return maze_state

    @override(CoreEnv)
    def seed(self, seed: int) -> None:
        """Intercept ``CoreEnv.seed``"""
        self.env.seed(seed)

    @override(CoreEnv)
    def is_actor_done(self) -> bool:
        """Intercept ``CoreEnv.is_actor_done``"""
        return False

    @override(CoreEnv)
    def actor_id(self) -> ActorID:
        """Intercept ``CoreEnv.actor_id``"""
        return ActorID(step_key=0, agent_id=0)

    @property
    @override(CoreEnv)
    def agent_counts_dict(self) -> Dict[StepKeyType, int]:
        """Single policy, single agent env."""
        return {0: 1}

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'GymCoreEnv') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""

        # clone core env maze state
        self._maze_state = deepcopy(env._maze_state)

        # clone environment state
        parent_target_env = None
        parent_source_env = None
        target_env = self.env
        source_env = env.env
        while hasattr(target_env, "env"):
            assert hasattr(source_env, "env")

            # copy state of time limit wrapper
            if isinstance(target_env, TimeLimit):
                assert isinstance(source_env, TimeLimit)
                target_env._max_episode_steps = source_env._max_episode_steps
                target_env._elapsed_steps = source_env._elapsed_steps

            parent_target_env = target_env
            target_env = target_env.env
            parent_source_env = source_env
            source_env = source_env.env
            assert isinstance(source_env, target_env.__class__)

        # clone state of classic control environments
        control_envs = (CartPoleEnv, MountainCarEnv, Continuous_MountainCarEnv, PendulumEnv, AcrobotEnv)
        if isinstance(target_env, control_envs):
            assert isinstance(source_env, control_envs)
            parent_target_env.env = deepcopy(parent_source_env.env)
        # clone state of atari environments
        elif AtariEnv and isinstance(target_env, AtariEnv):
            assert isinstance(source_env, AtariEnv)
            target_env.np_random = deepcopy(source_env.np_random)
            state = source_env.ale.cloneState()
            target_env.ale.restoreState(state)
        # reset is not supported yet
        else:
            raise RuntimeError(
                f"Cloning of {target_env.__class__} env not supported!"
                f"If working with an Atari env make sure all required dependencies are installed "
                f"(e.g., 'pip install gym[atari]')!")


class GymMazeEnv(MazeEnv):
    """Wraps a Gym env into a Maze environment.

    **Example**: *env = GymMazeEnv(env="CartPole-v0")*

    :param env: The gym environment to wrap or the environment id.
    """

    def __init__(self, env: Union[str, gym.Env]):
        if not isinstance(env, gym.Env):
            env = gym.make(env)

        super().__init__(
            core_env=GymCoreEnv(env),
            action_conversion_dict={0: GymActionConversion(env=env)},
            observation_conversion_dict={0: GymObservationConversion(env=env)})


def make_gym_maze_env(name: str) -> GymMazeEnv:
    """Initializes a :class:`~maze.core.wrappers.maze_gym_env_wrapper.GymMazeEnv` by registered Gym env name (id).

    :param name: The name (id) of a registered Gym environment.
    :return: The instantiated environment.
    """
    return GymMazeEnv(gym.make(name))
