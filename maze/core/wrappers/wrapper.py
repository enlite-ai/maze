"""Extension of gym Wrapper to support environment interfaces"""
import logging
import time
from abc import abstractmethod, ABC
from typing import Generator, TypeVar, Generic, Type, Union, Dict, Tuple, Any, Optional, List

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.environment_context import EnvironmentContext
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.log_events.env_profiling_events import EnvProfilingEvents

EnvType = TypeVar("EnvType")

logger = logging.getLogger('WRAPPER')
logger.setLevel(logging.INFO)


class Wrapper(Generic[EnvType], SimulatedEnvMixin, ABC):
    """
    A transparent environment Wrapper that works with any manifestation of :class:`~maze.core.env.base_env.BaseEnv`.
    It is intended as drop-in replacement for gymnasium.core.Wrapper.

    Gym Wrappers elegantly expose methods and attributes of all nested envs. However wrapping destroys the class
    hierarchy, querying the base classes is not straight-forward. This environment wrapper fixes the behaviour
    of isinstance() for arbitrarily nested wrappers.

    Suppose we want to check the base class:


        class MyGymWrapper(Wrapper[gymnasium.Env]):
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

    gymnasium.core.Wrapper assumes the existence of certain attributes (action_space, observation_space, reward_range,
    metadata) and duplicates these attributes. This behaviour is unnecessary, because __getattr__ makes these
    members of the inner environment transparently available anyway.
    """

    def __init__(self, env: EnvType):
        """Avoid calling this constructor directly, use :method:`wrap` instead."""
        self.env = env

        base_classes = self._base_classes(env)
        base_classes = list(dict.fromkeys(base_classes))

        # Setup profiling events in case there is a core env present in the underlying wrapper stack. Otherwise it is
        # disabled.
        self.log_profiling_events = True
        if not hasattr(self, 'core_env'):
            self.log_profiling_events = False

        self.__profiling_events = None
        if self.log_profiling_events:
            self.__profiling_events = self.core_env.context.event_service.create_event_topic(EnvProfilingEvents)

        # Each wrapper has it own profiling time
        self._last_profiling_time = 0.0
        # In case a skipping wrapper is used, this timing needs to be dealt with separately.
        self._cumulative_time_for_skipping_wrapper = 0
        # Used as a placeholder to keep track of the full wrapper stack used.
        self._full_wrapper_stack: List[str] = []

        # Notify the sub-wrappers with the full stack of wrappers uses.
        self.notify_full_wrapper_stack(self.get_full_wrapper_stack())
        self.last_step_id = None

        # we create a new Class type, that implements all base classes of the wrapping chain. This class is never
        # actually instantiated.
        class _FakeClass(*base_classes):
            pass

        self.fake_class = _FakeClass

        # wrap step function in callbacks (i.e., replace it with self.step_with_callbacks call)
        # env context is necessary to correctly detect when callbacks should be triggered
        if hasattr(env, "context") and isinstance(env.context, EnvironmentContext):
            self.__step = self.step
            self.step = self.step_with_callbacks

    def step_with_callbacks(self, *args, **kwargs) -> Any:
        """A wrapper for the env.step function. Checks whether callbacks for this step have already been process
        (i.e., detects whether this is the outermost wrapper). Triggers the post-step callbacks if required."""
        self._first_step_started = True

        if (self.context.current_wrapper_pos is not None and
                self.position_in_wrapper_stack() >= self.context.current_wrapper_pos):
            if self.log_profiling_events:
                self._record_profiling_events()
            self.context.current_wrapper_pos = None

        # check if new step has been initiated before fully completing the previous one (= step-skipping etc.)
        if self.context.step_in_progress and not self.context.step_is_initiating:
            # we are starting a second step from the middle of the wrapper hierarchy
            # without having fully completed the previous step (as is done e.g. during step-skipping)
            # => call the post-step callbacks here to allow clearing events etc. between skipped steps
            self.context.run_post_step_callbacks()
            self.context.step_is_initiating = True

        # if the step has not been marked as in progress yet, we know that this is the outer-most wrapper in the stack
        is_outermost_wrapper = not self.context.step_in_progress
        if is_outermost_wrapper:
            self.context.step_in_progress = True
            self.context.step_is_initiating = True
            # once pre-step callbacks are supported, they can be called here

        start_time = time.time()

        # now do the step (env step + any functionality this wrapper implements on top of that)
        return_value = self.__step(*args, **kwargs)

        self._last_profiling_time = time.time() - start_time

        # if this is the outer-most wrapper, mark the step as done and trigger post-step callbacks
        if is_outermost_wrapper:
            self.context.step_in_progress = False
            if self.log_profiling_events:
                self._record_profiling_events()
            self.context.run_post_step_callbacks()
            self.context.current_wrapper_pos = None
        else:
            self.context.current_wrapper_pos = self.position_in_wrapper_stack()

        return return_value

    def _record_profiling_events(self) -> None:
        """Record the profiling events for each wrapper as the abs wall time and percentage of the full step."""

        # First check if skipping was used. If so, this time needs to be subtracted from the recorded profiling time.
        skipping_cum_time = 0
        if not self.context.step_in_progress and self.is_actual_outermost_wrapper():
            skipping_cum_time = self.skipping_happened_since_last_call_to_this_wrapper()
            if skipping_cum_time > self._last_profiling_time:
                skipping_cum_time = 0

        # Total time with fallback for extrem fast envs
        total_time = max(self._last_profiling_time - skipping_cum_time, 1e-6)
        self.__profiling_events.full_env_step_time(total_time)

        # Record the profiling time for each of the wrappers used.
        current_env = self
        step_down_env = self.env
        recorded_wrappers = self._full_wrapper_stack[:]
        while self.is_wrapper(step_down_env):
            current_run_time = current_env._last_profiling_time - step_down_env._last_profiling_time
            if step_down_env._cumulative_time_for_skipping_wrapper > 0:
                current_run_time = current_run_time - skipping_cum_time
            name = type(current_env).__name__
            recorded_wrappers.remove(name)
            self.__profiling_events.wrapper_step_time(wrapper_name=name, time=current_run_time,
                                                      per=current_run_time / total_time)
            current_env = step_down_env
            step_down_env = current_env.env

        # Record 0 for all wrappers that where not called (due to skipping).
        for wr in recorded_wrappers:
            self.__profiling_events.wrapper_step_time(wrapper_name=wr, time=0, per=0)

        self.record_skipping_time(list(recorded_wrappers), total_time)

        # Now current env is maze env and step_down env is core env.
        maze_env_run_time = current_env._last_profiling_time
        core_env_run_time = current_env.profiling_times['core_env']
        maze_env_run_time_diff = (
                maze_env_run_time - core_env_run_time - current_env.profiling_times['observation_conversion'] -
                current_env.profiling_times['action_conversion'])
        self.__profiling_events.maze_env_step_time(time=maze_env_run_time_diff, per=maze_env_run_time_diff / total_time)
        self.__profiling_events.core_env_step_time(time=core_env_run_time, per=core_env_run_time / total_time)
        self.__profiling_events.observation_conv_time(
            time=current_env.profiling_times['observation_conversion'],
            per=current_env.profiling_times['observation_conversion'] / total_time)
        self.__profiling_events.action_conv_time(time=current_env.profiling_times['action_conversion'],
                                                 per=current_env.profiling_times['action_conversion'] / total_time)

        # In case the reserved dictionary is used in the core env, retrieve the values and log them.
        if hasattr(self, '_investigate_step_function_parts'):
            for key, value in self._investigate_step_function_parts.items():
                self.__profiling_events.investigate_time(name=key, time=value, per=value / core_env_run_time)

    @property
    def __class__(self):
        """
        Making `isinstance()` work on this instance is done by overriding __class__, a hack that is also used by
        the Mock library and therefore should be stable.
        """
        return self.fake_class

    def __getattr__(self, name):
        """If the attribute is not available directly on this wrapper, query the wrapped env below."""
        return getattr(self.env, name)

    @property
    def is_initialized(self) -> bool:
        """True if the initialization has been completed (i.e., after the `__init__` and `wrap` methods
        have finished). The `_is_initialized` flag is set at the end of the `wrap` method."""
        return "_is_initialized" in self.__dict__ and self.__dict__["_is_initialized"]

    def __setattr__(self, name, value):
        """Complementary to the overridden `__getattr__` above. Outside of wrapper initialization, if the attribute
        we are assigning is not available on this wrapper, check the wrapped environment below instead."""

        # Set the attribute directly on this wrapper only (1) during initialization, or (2) if this wrapper already
        # has an attribute of this name
        if not self.is_initialized or name in self.__dict__ or name == "env":
            self.__dict__[name] = value
            return

        # If none of the conditions above hold, attempt to set the attribute on the wrapped env,
        # falling back to self if the env is not available.
        env = self.__dict__.get("env", None)
        if not env:
            self.__dict__[name] = value
            return

        return setattr(env, name, value)

    @staticmethod
    def is_wrapper(env: Any) -> bool:
        """Return true if the given env is a wrapper."""
        return isinstance(env, Wrapper)

    def get_full_wrapper_stack(self) -> List[str]:
        """Get a list of the full wrapper stack.

        :return: A list of wrapper from this wrapper going to the maze env.
        """
        wrapper_stack = []
        current_env = self
        while self.is_wrapper(current_env):
            wrapper_stack.append(type(current_env).__name__)
            current_env = current_env.env
        # Since the MazeEnv inherits from wrapper is counts as a wrapper and needs to be excluded.
        return wrapper_stack[:-1]

    def notify_full_wrapper_stack(self, wrapper_names: List[str]):
        """Pass the full list of wrappers used down to every wrapper, such that each wrapper has the full list.

        :param wrapper_names: List of wrapper names used.
        """
        self._full_wrapper_stack = wrapper_names
        if self.is_wrapper(self.env):
            self.env.notify_full_wrapper_stack(wrapper_names)

    def position_in_wrapper_stack(self) -> int:
        """Return the current position in the wrapper stack as an integer."""
        recorded_wrappers = self._full_wrapper_stack[:]
        if type(self).__name__ not in recorded_wrappers:
            return len(recorded_wrappers)
        else:
            return recorded_wrappers.index(type(self).__name__)

    def skipping_happened_since_last_call_to_this_wrapper(self) -> float:
        """Check if skipping has happen since the last call to this wrapper."""
        cur_env = self
        while self.is_wrapper(cur_env):
            if cur_env._cumulative_time_for_skipping_wrapper > 0:
                return cur_env._cumulative_time_for_skipping_wrapper
            cur_env = cur_env.env
        return 0

    def is_actual_outermost_wrapper(self) -> bool:
        """Return true if this is the actual outermost wrapper, this is necessary since skipping in reset is sometimes
        possible."""
        recorded_wrappers = self._full_wrapper_stack[:]
        current_env = self
        while self.is_wrapper(current_env.env):
            name = type(current_env).__name__
            recorded_wrappers.remove(name)
            current_env = current_env.env
        return len(recorded_wrappers) == 0

    def record_skipping_time(self, recorded_wrappers: List[str], total_time: float) -> None:
        """Record the skipping time if necessary.

        :param recorded_wrappers: List of wrapper names used.
        :param total_time: Total time taken to execute this wrapper.
        """
        if len(recorded_wrappers) > 0:
            self._cumulative_time_for_skipping_wrapper += total_time
        else:
            cur_env = self
            while self.is_wrapper(cur_env):
                cur_env._cumulative_time_for_skipping_wrapper = 0
                cur_env = cur_env.env

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
        """Creation method providing appropriate type hints. Preferred method to construct the wrapper compared to calling
        the class constructor directly.

        Note: If you are overriding this method, do not forget to set the `_is_initialized` flag at the end.

        :param env: The environment to be wrapped
        :param kwargs: Arguments to be passed on to wrapper's constructor.
        :return: A newly created wrapper instance. Since we want to allow sub-classes to use .wrap() without having to
                 reimplement them and still facilitate proper typing hints, we use a generic to represent the type
                 of cls. See:
                 https://stackoverflow.com/questions/39205527/can-you-annotate-return-type-when-value-is-instance-of-cls/39205612#39205612
                 on why/how to use this to indicate that an instance of cls is returned.
        """
        instance = cls(env, **kwargs)
        instance._is_initialized = True  # Set the flag at the end of the initialization
        return instance

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

    @override(SimulatedEnvMixin)
    def serialize_state(self) -> Any:
        """Serialize the current env state and return an object that can be used to deserialize the env again.
        NOTE: The method is optional.
        """
        return self.env.serialize_state()

    @override(SimulatedEnvMixin)
    def deserialize_state(self, serialized_state: Any) -> None:
        """Deserialize the current env from the given env state."""
        self.env.deserialize_state(serialized_state)

    def noop_action(self) -> ActionType:
        """Gets the noop_action for this environment.

        By default, this call is just forwarded down the wrapper stack until it reaches MazeEnv,
        which attempts to get a noop_action from its action conversion interface. However,
        if your wrapper somehow manipulates the action format (e.g., by turning the env
        into a multi-step one), you should override this method and provide a compatible implementation.
        """
        return self.env.noop_action()


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
        if not done:
            observation = self.observation(observation)
        return observation, reward, done, info

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
