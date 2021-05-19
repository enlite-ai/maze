"""Extension of gym Wrapper to support environment interfaces"""
from abc import abstractmethod, ABC
from typing import Generator, TypeVar, Generic, Type, Union, Dict, Tuple, Any, Optional, Callable

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.environment_context import EnvironmentContext
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin

EnvType = TypeVar("EnvType")


class Wrapper(Generic[EnvType], SimulatedEnvMixin, ABC):
    """
    A transparent environment Wrapper that works with any manifestation of :class:`~maze.core.env.base_env.BaseEnv`.
    It is intended as drop-in replacement for gym.core.Wrapper.

    Gym Wrappers elegantly expose methods and attributes of all nested envs. However wrapping destroys the class
    hierarchy, querying the base classes is not straight-forward. This environment wrapper fixes the behaviour
    of isinstance() for arbitrarily nested wrappers.

    Suppose we want to check the base class:

        class MyGymWrapper(Wrapper[gym.Env]):
            ...

        # construct an env and wrap it
        env = MyEnv()
        env = MyGymWrapper(env)

        # this assertion fails
        assert isinstance(env, MyEnv) == True

    TypingWrapper makes `isinstance()` work as intuitively expected:

        # this time use MyWrapper, which is derived from this Wrapper class
        env = MyEnv()
        env = MyWrapper(env)

        # now the assertions hold
        assert isinstance(env, MyEnv) == True
        assert isinstance(env, MyWrapper) == True


    Note:

    gym.core.Wrapper assumes the existence of certain attributes (action_space, observation_space, reward_range,
    metadata) and duplicates these attributes. This behaviour is unnecessary, because __getattr__ makes these
    members of the inner environment transparently available anyway.

    :param keep_inner_hooks: Set this to True if the Wrapper calls additional step() functions, as e.g. in a frame
                             skip scenario. In that case the hooks are not only invoked at the root of the wrapper
                             hierarchy, but additionally also by this wrapper.
    """

    def __init__(self, env: EnvType, keep_inner_hooks=False):
        self.env = env
        self.keep_inner_hooks = keep_inner_hooks

        base_classes = self._base_classes(env)
        base_classes = list(dict.fromkeys(base_classes))

        # we create a new Class type, that implements all base classes of the wrapping chain. This class is never
        # actually instantiated.
        class _FakeClass(*base_classes):
            pass

        self.fake_class = _FakeClass

        # intercept the first step function call in the wrapper hierarchy for
        # a reliable place to trigger the event system reset
        if hasattr(env, "context") and isinstance(env.context, EnvironmentContext):
            assert isinstance(env.context, EnvironmentContext)
            self._register_hook("step", env.context.pre_step)

    @classmethod
    def _create_proxy(cls, original_fn: Callable, pre_fn: Callable) -> Callable:
        def _proxy_function(*args, **kwargs) -> Any:
            pre_fn()
            return original_fn(*args, **kwargs)

        return _proxy_function

    def _register_hook(self, name: str, pre_fn: Callable) -> None:
        """Intercept the given list of methods (e.g. the "step" method) and trigger the pre function.

        Pre-functions are usually only invoked on the outermost Wrapper of the entire wrapper - environment stack.
        But there is no guarantee, that hooks will be invoked exactly once per step, see the `keep_inner_hooks` flag.

        :param name: The method as function name, which will be overwritten by the proxy function.
        """
        # We register hooks only for the outermost wrapper. In case this wrapper was added to an existing wrapper
        # stack, we need to remove the hooks from the previous wrapper.
        if isinstance(self.env, Wrapper) and not self.keep_inner_hooks:
            self.env.unregister_hook(name)

        original_fn = getattr(self, name, None)
        proxy_function = self._create_proxy(original_fn, pre_fn)
        if proxy_function:
            # safe the original function
            setattr(self, f"__{name}", original_fn)
            # make the proxy function the new step function
            setattr(self, name, proxy_function)

    def unregister_hook(self, method_name: str) -> None:
        """Remove the method interception hook, by restoring the original method functions.

        :param method_name: The method as function name, which will be restored.
        """
        original_fn = getattr(self, f"__{method_name}", None)
        if original_fn:
            setattr(self, method_name, original_fn)

    @property
    def __class__(self):
        """
        Making `isinstance()` work on this instance is done by overriding __class__, a hack that is also used by
        the Mock library and therefore should be stable.
        """
        return self.fake_class

    def __getattr__(self, name):
        return getattr(self.env, name)

    @classmethod
    def _base_classes(cls, env: BaseEnv) -> Generator[BaseEnv, None, None]:
        # start collecting all base classes by traversing the wrapping hierarchy
        inner_env = env
        while True:
            yield type(inner_env)

            # rely on duck typing
            if not hasattr(inner_env, "env"):
                break

            inner_env = inner_env.env

        # return the bases of the wrapper class as well
        for base in cls.__bases__:
            # skip the Generic base
            if base is Generic:
                continue

            yield base

    # Type to be returned when using the inheritable .wrap() method. See:
    # https://stackoverflow.com/questions/39205527/can-you-annotate-return-type-when-value-is-instance-of-cls/39205612#39205612
    WrapperType = TypeVar("WrapperType", bound='Wrapper')

    # specific env type, ensures that we don't use details about our type
    T = TypeVar("T")

    @classmethod
    def wrap(cls: Type[WrapperType], env: T, **kwargs) -> Union[T, WrapperType]:
        """
        Creation method providing appropriate type hints. Preferred method to construct the wrapper compared to calling
        the class constructor directly.
        :param env: The environment to be wrapped
        :param kwargs: Arguments to be passed on to wrapper's constructor.
        :return A newly created wrapper instance. Since we want to allow sub-classes to use .wrap() without having to
        reimplement them and still facilitate proper typing hints, we use a generic to represent the type of cls. See
        https://stackoverflow.com/questions/39205527/can-you-annotate-return-type-when-value-is-instance-of-cls/39205612#39205612
        on why/how to use this to indicate that an instance of cls is returned.
        """

        return cls(env, **kwargs)

    # implementing the interfaces below is optional for use cases where you actually need them

    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Convert MazeState and MazeAction back into raw action and observation.

        This method is mostly used when working with trajectory data, e.g. for imitation learning. As part
        of trajectory data, MazeState and MazeActions are recorded. For imitation learning, they then need to be
        converted
        to raw observations and actions in the desired format (i.e. using all the required wrappers etc.)

        The conversion is done by first transforming the MazeState and MazeAction using the space interfaces in
        MazeEnv, and then running them through the entire wrapper stack ("back up").

        Both the MazeState and the MazeAction on top of it are converted as part of this single method,
        as some wrappers (mostly multi-step ones) need them both together (to be able to split them
        into observations and actions taken in different sub-steps). If you are not using multi-step wrappers,
        you don't need to convert both MazeState and MazeAction, you can pass in just one of them. Not all wrappers
        have to support this though.

        See below for an example implementation.

        Note: The conversion of MazeState to observation is in the "natural" direction, how it takes place when stepping
        the env. This is not true for the MazeAction to action conversion -- when stepping the env, actions are
        converted to MazeActions, whereas here the MazeAction needs to be converted back into the "raw" action (i.e.
        in reverse direction).

        (!) Attention: In case that there are some stateful wrappers in the wrapper stack (e.g. a wrapper stacking
        observations from previous steps), you should ensure that (1) the first_step_in_episode flag is passed
        to this function correctly and (2) that all states and MazeActions are converted in order -- as they happened
        during the recorded episode.

        :param maze_state: MazeState to convert. If none, only MazeAction will be converted
                          (not all wrappers support this).
        :param maze_action: MazeAction (the one following the state given as the first param). If none, only MazeState
                          will be converted (not all wrappers support this, some need both).
        :param first_step_in_episode: True if this is the first step in the episode. Serves to notify stateful wrappers
                                      (e.g. observation stacking) that they should reset their state.
        :return: observation and action dictionaries (keys are IDs of sub-steps)
        """

        # First pass maze_state and maze_action down the wrapper stack:
        #
        # obs_dict, act_dict = self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)
        #
        # if first_step_in_episode:
        #   # If the wrapper keeps any state (e.g. stack of past observations), reset it
        #
        # Now do the conversion and return the observation and action dicts back up:
        # ...
        # return converted_obs_dict, converted_act_dict

        raise NotImplementedError

    @override(SimulatedEnvMixin)
    def clone_from(self, env: EnvType) -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`.

        Note: implementing this method is required for stateful environment wrappers.
        """
        raise NotImplementedError


class ObservationWrapper(Wrapper[EnvType], ABC):
    """A Wrapper with typing support modifying the environments observation."""

    @override(BaseEnv)
    def reset(self) -> Any:
        """Intercept ``BaseEnv.reset`` and map observation."""
        observation = self.env.reset()
        return self.observation(observation)

    def step(self, action) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Intercept ``BaseEnv.step`` and map observation."""
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    @abstractmethod
    def observation(self, observation: Any) -> Any:
        """Observation mapping method."""

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Convert the observations, keep actions the same."""
        obs_dict, act_dict = self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)
        if obs_dict is not None:
            obs_dict = {policy_id: self.observation(obs) for policy_id, obs in obs_dict.items()}
        return obs_dict, act_dict


class ActionWrapper(Wrapper[EnvType], ABC):
    """A Wrapper with typing support modifying the agents action."""

    @override(BaseEnv)
    def step(self, action) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Intercept ``BaseEnv.step`` and map action."""
        return self.env.step(self.action(action))

    @abstractmethod
    def action(self, action: Any) -> Any:
        """Abstract action mapping method."""

    @abstractmethod
    def reverse_action(self, action: Any) -> Any:
        """Abstract action reverse mapping method."""

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Reverse the actions, keep the observations the same."""
        obs_dict, act_dict = self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)
        if act_dict is not None:
            act_dict = {policy_id: self.reverse_action(action) for policy_id, action in act_dict.items()}
        return obs_dict, act_dict


class RewardWrapper(Wrapper[EnvType], ABC):
    """A Wrapper with typing support modifying the reward before passed to the agent."""

    def step(self, action) -> Tuple[Any, Any, bool, Dict[Any, Any]]:
        """Intercept ``BaseEnv.step`` and map rewards."""
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    @abstractmethod
    def reward(self, reward: Any) -> Any:
        """Reward mapping method."""

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Keep both actions and observation the same."""
        return self.env.get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)
