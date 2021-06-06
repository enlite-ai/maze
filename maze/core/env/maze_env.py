"""A general implementation for wrapping a core environment in a (gym style) environment.

Implementations based on core environments (:class:`~.core_env.CoreEnv`) come with a set of individually
tailored MazeState and MazeAction objects. Having no global assumptions on the structure of these objects allows for
optimal representations and minimal, clean environment logic.

RL training algorithms require a more rigid representation. To that end :class:`MazeEnv` wraps the environment as
gym-compatible environment in a reusable form, by utilizing mappings from the MazeState to the observations space and
from the MazeAction to the action space.
"""
from typing import Any, Tuple, Dict, Iterable, Optional, Union, TypeVar, Generic

import gym
import numpy as np

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionConversionInterface, ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.core_env import CoreEnv
from maze.core.env.event_env_mixin import EventEnvMixin
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationConversionInterface, ObservationType
from maze.core.env.recordable_env_mixin import RecordableEnvMixin
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env import StructuredEnv, StepKeyType, ActorID
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.env.time_env_mixin import TimeEnvMixin
from maze.core.events.event_record import EventRecord
from maze.core.log_events.kpi_calculator import KpiCalculator
from maze.core.log_events.monitoring_events import RewardEvents
from maze.core.rendering.renderer import Renderer
from maze.core.wrappers.wrapper import Wrapper

CoreEnvType = TypeVar("CoreEnvType")


class MazeEnv(Generic[CoreEnvType], Wrapper[CoreEnvType], StructuredEnv, StructuredEnvSpacesMixin, EventEnvMixin,
              RecordableEnvMixin, TimeEnvMixin):
    """Base class for (gym style) environments wrapping a core environment and defining state and execution interfaces.
    The aim of this class is to provide reusable functionality across different gym environments.
    This functionality comprises for example the reset-function, the step-function or the render-function.

    :param core_env: Core environment.
    :param action_conversion_dict: A dictionary with action conversion interface implementation
                                     and policy names as keys.
    :param observation_conversion_dict: A dictionary with observation conversion interface implementation
                                      and policy names as keys.
    """

    def __init__(self,
                 core_env: CoreEnv,
                 action_conversion_dict: Dict[Union[str, int], ActionConversionInterface],
                 observation_conversion_dict: Dict[Union[str, int], ObservationConversionInterface]):
        self.core_env = core_env
        """wrapped :class:`~.core_env.CoreEnv`"""

        self.maze_env = self
        """direct access to the maze env (useful to bypass the wrapper hierarchy)"""

        self.action_conversion_dict = action_conversion_dict
        """The action conversion mapping used by this env."""
        self.observation_conversion_dict = observation_conversion_dict
        """The observation conversion mapping used by this env."""

        self._init_spaces()

        super().__init__(self.core_env)

        # this is required to stay compatible with the gym.Env interface
        self.metadata = {'render.modes': []}
        """Only there to be compatible with gym.core.Env"""
        self.reward_range = (-float('inf'), float('inf'))
        """A tuple (reward min value, reward max value) to be compatible with gym.core.Env"""
        self.spec = None
        """Only there to be compatible with gym.core.Env"""

        # last MazeAction taken for trajectory data recording
        self.last_maze_action = None

        # last observation, captured immediately after the observation_conversion mapping
        self.observation_original = None

        # create event topics
        self.reward_events = self.core_env.context.event_service.create_event_topic(RewardEvents)

        # Check if the underlying core env has only single sub-step
        agent_counts = self.core_env.agent_counts_dict
        self.is_single_substep_env = len(agent_counts) == 1 and sum(agent_counts.values()) == 1

    @override(BaseEnv)
    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, Dict[Any, Any]]:
        """Take environment step (see :func:`CoreEnv.step <maze.core.env.core_env.CoreEnv.step>` for details).

        :param action: the action the agent wants to take.
        :return: observation, reward, done, info
        """

        # first, take step without observation
        reward, done, info = self._step_core_env(action)

        # convert state to observation
        maze_state = self.core_env.get_maze_state()
        self.observation_original = observation = self.observation_conversion.maze_to_space(maze_state)

        return observation, reward, done, info

    @override(BaseEnv)
    def reset(self) -> ObservationType:
        """Resets the environment and returns the initial observation.

        :return: the initial observation after resetting.
        """
        self.core_env.context.reset_env_episode()
        maze_state = self.core_env.reset()

        self.observation_original = observation = self.observation_conversion.maze_to_space(maze_state)
        for key, value in observation.items():
            assert not (isinstance(value, np.ndarray) and value.dtype == np.float64), \
                   f"observation contains numpy arrays with float64, please convert observation '{key}' to float32"

        return observation

    @override(BaseEnv)
    def seed(self, seed: Any) -> None:
        """forward call to :attr:`self.core_env <core_env>`
        """
        return self.core_env.seed(seed)

    @override(BaseEnv)
    def close(self) -> None:
        """forward call to :attr:`self.core_env <core_env>`
        """
        return self.core_env.close()

    @override(CoreEnv)
    def get_step_events(self) -> Iterable[EventRecord]:
        """forward call to :attr:`self.core_env <core_env>`
        """
        return self.core_env.get_step_events()

    @override(CoreEnv)
    def get_kpi_calculator(self) -> Optional[KpiCalculator]:
        """forward call to :attr:`self.core_env <core_env>`
        """
        return self.core_env.get_kpi_calculator()

    @override(RecordableEnvMixin)
    def get_maze_state(self) -> MazeStateType:
        """Return current State object for the core env for trajectory recording."""
        return self.core_env.get_maze_state()

    @override(RecordableEnvMixin)
    def get_episode_id(self) -> str:
        """Return the ID of current episode (the ID changes on env reset)."""
        return self.core_env.context.episode_id

    @override(RecordableEnvMixin)
    def get_maze_action(self) -> MazeActionType:
        """Return last MazeAction object for trajectory recording."""
        return self.last_maze_action

    @override(RecordableEnvMixin)
    def get_renderer(self) -> Renderer:
        """Return the renderer exposed by the underlying core env."""
        return self.core_env.get_renderer()

    @property
    @override(StructuredEnvSpacesMixin)
    def action_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Space]:
        """Policy action spaces as dict."""
        return self._action_spaces

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_spaces_dict(self) -> Dict[Union[int, str], gym.spaces.Space]:
        """Policy observation spaces as dict."""
        return self._observation_spaces

    @override(TimeEnvMixin)
    def get_env_time(self) -> int:
        """Forward the call to :attr:`self.core_env <core_env>`"""
        return self.core_env.get_env_time()

    def _init_spaces(self) -> None:
        """Initialize observation and action space.
        """
        self._observation_spaces = {k: obs_conv.space() for k, obs_conv in self.observation_conversion_dict.items()}
        self._action_spaces = {k: act_conv.space() for k, act_conv in self.action_conversion_dict.items()}

    @override(StructuredEnv)
    def actor_id(self) -> ActorID:
        """forward call to :attr:`self.core_env <core_env>`"""
        return self.core_env.actor_id()

    @override(StructuredEnv)
    def is_actor_done(self) -> bool:
        """forward call to :attr:`self.core_env <core_env>`"""
        return self.core_env.is_actor_done()

    @property
    @override(StructuredEnv)
    def agent_counts_dict(self) -> Dict[StepKeyType, int]:
        """forward call to :attr:`self.core_env <core_env>`"""
        return self.core_env.agent_counts_dict

    @override(StructuredEnv)
    def get_actor_rewards(self) -> Optional[np.ndarray]:
        """forward call to :attr:`self.core_env <core_env>`"""
        return self.core_env.get_actor_rewards()

    @property
    def action_space(self):
        """Keep this env compatible with the gym interface by returning the
        action space of the current policy."""
        policy_id, actor_id = self.core_env.actor_id()
        return self.action_spaces_dict[policy_id]

    @property
    def observation_space(self):
        """Keep this env compatible with the gym interface by returning the
        observation space of the current policy."""
        policy_id, actor_id = self.core_env.actor_id()
        return self.observation_spaces_dict[policy_id]

    @property
    def action_conversion(self):
        """Return the action conversion mapping for the current policy."""
        policy_id, actor_id = self.core_env.actor_id()
        return self.action_conversion_dict[policy_id]

    @property
    def observation_conversion(self):
        """Return the state to observation mapping for the current policy."""
        policy_id, actor_id = self.core_env.actor_id()
        return self.observation_conversion_dict[policy_id]

    @override(Wrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType], first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Convert MazeState and MazeAction back into observations and actions using the space conversion interfaces.

        :param maze_state: State of the environment
        :param maze_action: MazeAction (the one following the state given as the first param)
        :param first_step_in_episode: True if this is the first step in the episode.
        :return: observation and action dictionaries (keys are substep_ids)
        """

        if maze_state is not None:
            observation_dict = {
                policy_id: obs_conv.maze_to_space(maze_state)
                for policy_id, obs_conv in self.observation_conversion_dict.items()
            }
        else:
            observation_dict = None

        if maze_action is not None:
            action_dict = {
                policy_id: act_conv.maze_to_space(maze_action)
                for policy_id, act_conv in self.action_conversion_dict.items()
            }
        else:
            action_dict = None

        return observation_dict, action_dict

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'MazeEnv') -> None:
        """Reset the maze env to the state of the provided env.

        Note, that it also clones the CoreEnv and its member variables including environment context.

        :param env: The environment to clone from.
        """
        self.core_env.clone_from(env.core_env)

        self.core_env.context.clone_from(env.core_env.context)
        self.core_env.reward_aggregator.clone_from(env.core_env.reward_aggregator)

    def _step_core_env(self, action: ActionType) -> Tuple[float, bool, Dict[Any, Any]]:
        """Take environment step without converting the state into and observation.

        :param action: the action the agent wants to take.
        :return: reward, done, info.
        """
        last_env_time = self.get_env_time()

        # compile action object
        maze_state = self.core_env.get_maze_state()
        maze_action = self.action_conversion.space_to_maze(action, maze_state)

        # take environment step
        maze_state, reward, done, info = self.core_env.step(maze_action)

        if self.is_single_substep_env or last_env_time != self.get_env_time():
            # aggregate to scalar reward (if necessary)
            if self.core_env.reward_aggregator:
                reward = self.core_env.reward_aggregator.to_scalar_reward(reward)

            # reward captured immediately after the reward aggregation
            self.reward_events.reward_original(value=reward)

            # record the last MazeAction
            self.last_maze_action = maze_action

            # For single sub-step core envs which do not manage their env time:
            # Schedule a new step and clear event logs automatically
            if self.is_single_substep_env and last_env_time == self.get_env_time():
                self.core_env.context.increment_env_step()

            # ensure that all reward aggregator are cleared
            self.core_env.context.event_service.clear_pubsub()

        return reward, done, info
