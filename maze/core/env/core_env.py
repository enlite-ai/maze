"""Core environment interfaces.

Core environments form the basis for actual RL trainable environments (e.g. gym.Envs). Instead of operating with
observations and actions they operate with MazeStates (:mod:`~.maze_state`) and MazeActions
(:mod:`~.maze_action`). This design choice give much more freedom when for example implementing
heuristic policies for a specific environment (It is much easier to implement a heuristics
given a clean state representation compared to a dictionary action space of machine readable arrays).
"""
from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Union, Iterable, Optional

import numpy as np

from maze.core.annotations import override
from maze.core.env.environment_context import EnvironmentContext
from maze.core.env.event_env_mixin import EventEnvMixin
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.reward import RewardAggregatorInterface
from maze.core.env.serializable_env_mixin import SerializableEnvMixin
from maze.core.env.structured_env import StructuredEnv, StepKeyType, ActorID
from maze.core.env.time_env_mixin import TimeEnvMixin
from maze.core.events.event_record import EventRecord
from maze.core.log_events.kpi_calculator import KpiCalculator
from maze.core.rendering.renderer import Renderer


class CoreEnv(StructuredEnv, EventEnvMixin, SerializableEnvMixin, TimeEnvMixin, ABC):
    """Interface definition for core environments forming the basis for actual RL trainable environments.
    """

    def __init__(self):
        self.context = EnvironmentContext()
        self.reward_aggregator: Optional[RewardAggregatorInterface] = None

    @abstractmethod
    @override(StructuredEnv)
    def step(self, maze_action: MazeActionType) -> \
            Tuple[MazeStateType, Union[float, np.ndarray, Any], bool, Dict[Any, Any]]:
        """Environment step function.

        Note: If your core environment is structured, you should call
        :func:`maze.core.env.environment_context.EnvironmentContext.increment_env_step()`
        once the structured step is done, so that the env time is incremented and events/stats cleared.

        :param maze_action: Environment MazeAction to take.
        :return: state, reward, done, info
        """

    @abstractmethod
    @override(StructuredEnv)
    def reset(self) -> MazeStateType:
        """Reset the environment and return initial state.

        :return: The initial state after resetting.
        """

    @abstractmethod
    @override(StructuredEnv)
    def seed(self, seed: int) -> None:
        """Sets the seed for this environment's random number generator(s).

        :param: seed: the seed integer initializing the random number generator.
        """

    @abstractmethod
    @override(StructuredEnv)
    def close(self) -> None:
        """Performs any necessary cleanup.
        """

    @abstractmethod
    def get_maze_state(self) -> MazeStateType:
        """Return current state of the environment.

        :return The same state as returned by reset().
        """

    @override(TimeEnvMixin)
    def get_env_time(self) -> int:
        """Return ID of the current step as env time."""
        return self.context.step_id

    @override(EventEnvMixin)
    def get_step_events(self) -> Iterable[EventRecord]:
        """Get all events recorded in the current step from the EventService.

        :return An iterable of the recorded events.
        """
        return self.context.event_service.iterate_event_records()

    @override(EventEnvMixin)
    def get_kpi_calculator(self) -> Optional[KpiCalculator]:
        """By default, Core Envs do not have to support KPIs."""
        return None

    @abstractmethod
    @override(SerializableEnvMixin)
    def get_serializable_components(self) -> Dict[str, Any]:
        """List components that should be serialized as part of trajectory data."""

    @abstractmethod
    def get_renderer(self) -> Renderer:
        """Return renderer instance that can be used to render the env.

        :return Renderer instance
        """

    @abstractmethod
    @override(StructuredEnv)
    def actor_id(self) -> ActorID:
        """Returns the currently executed actor. The id is unique only with
        respect to the policies (every policy has its own agent 0).

        Note that identities of done actors can not be reused in the same rollout.

        :return: The current actor, as a named tuple holding step_key and agent_id.
        """

    @abstractmethod
    @override(StructuredEnv)
    def is_actor_done(self) -> bool:
        """Returns True if the just stepped actor is done, which is different to the done flag of the environment.

        :return: True if the actor is done.
        """

    @property
    @abstractmethod
    @override(StructuredEnv)
    def agent_counts_dict(self) -> Dict[StepKeyType, int]:
        """Returns the maximum count of agents per sub-step that the environment features.

        For example:
          - For a vehicle-routing environment where 5 driver agents will get to act during sub-step 0,
            this method should return {0: 5}
          - For a two-step cutting environment where a piece is selected during sub-step 0 and then cut during
            sub-step 1 (with just one selection and cut happening in each step),
            this method should return {0: 1, 1: 1}
        """

    def clone_from(self, env: 'CoreEnv') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`.

        Cloning 'self.context' and 'self.reward_aggregator' is not required here anymore as it is already implemented
        in the :class:`~maze.core.env.maze_env.MazeEnv` as default behaviour.
        """
        raise NotImplementedError
