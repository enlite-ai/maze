"""Dummy structured (multi-agent, with two agents) core environment."""
import pickle
from typing import Tuple, Dict, Any, Optional

import gym
import numpy as np

from maze.core.annotations import override
from maze.core.env.core_env import CoreEnv
from maze.core.env.structured_env import StepKeyType, ActorID


class DummyStructuredCoreEnvironment(CoreEnv):
    """Dummy structured (multi-agent, with two agents) core environment.

    :param observation_space: The observation space to sample observations from.
    :param n_agents: The number of agents the env should have.
    """

    def __init__(self, observation_space: gym.spaces.space.Space, n_agents: int):
        super().__init__()
        self.observation_space = observation_space
        self.n_agents = n_agents
        self.current_agent = 0
        # Keep track of what actions have been taken
        self._current_path_id = 0

    @override(CoreEnv)
    def step(self, maze_action: Dict) -> Tuple[Dict[str, np.ndarray], float, bool, Optional[Dict]]:
        """Switch agents, increment env step after the second agent"""
        self.current_agent += 1
        action_hash = hash(tuple([tt if isinstance(tt, (int, np.int64)) else tuple(tt) for tt in maze_action.values()]))
        self._current_path_id += action_hash if action_hash >= 0 else - action_hash

        if self.current_agent % self.n_agents == 0:
            self.current_agent = 0
            self.context.increment_env_step()
            return self.get_maze_state(), 2, False, {}

        return self.get_maze_state(), 0, False, {}

    @override(CoreEnv)
    def get_maze_state(self) -> Dict[str, np.ndarray]:
        """Sample a random observation."""
        # Seed the observation space before sampling with the id of the current path in order to be seeding consistent
        self.observation_space.seed(self._current_path_id)
        return self.observation_space.sample()

    @override(CoreEnv)
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset current agent"""
        self.current_agent = 0
        self._current_path_id = 0
        return self.get_maze_state()

    @override(CoreEnv)
    def seed(self, seed: int) -> None:
        """No randomness in this env."""
        self.observation_space.seed(seed)

    @override(CoreEnv)
    def get_serializable_components(self) -> Dict[str, Any]:
        """No components required/available"""
        return {}

    @override(CoreEnv)
    def get_renderer(self) -> None:
        """No renderer available"""
        return None

    @override(CoreEnv)
    def actor_id(self) -> ActorID:
        """Single-step, two-agent environment"""
        return ActorID(step_key=0, agent_id=self.current_agent)

    @property
    @override(CoreEnv)
    def agent_counts_dict(self) -> Dict[StepKeyType, int]:
        """Single-step, two-agent environment"""
        return {0: self.n_agents}

    @override(CoreEnv)
    def is_actor_done(self) -> bool:
        """Actors are never done"""
        return False

    @override(CoreEnv)
    def get_actor_rewards(self) -> Optional[np.ndarray]:
        """Return reward of 1 for each actor as the last reward"""
        return np.array([1] * self.n_agents)

    @override(CoreEnv)
    def close(self) -> None:
        """Nothing to clean up"""
        pass

    def serialize_state(self) -> Any:
        """Serialize the current env state and return an object that can be used to deserialize the env again.
        """
        return pickle.dumps(
            [self.current_agent, self.n_agents, self.context.step_id, self.context.episode_id, self.observation_space,
             self._current_path_id])

    def deserialize_state(self, serialized_state: Any) -> None:
        """Deserialize the current env from the given env state."""
        self.current_agent, self.n_agents, self.context.step_id, self.context._episode_id, self.observation_space, \
            self._current_path_id = pickle.loads(serialized_state)

    @override(CoreEnv)
    def clone_from(self, env: 'CoreEnv') -> None:
        """Clone from the given env."""
        self.deserialize_state(env.serialize_state())
