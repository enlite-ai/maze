"""Episode records are composed of a chain of step records, each containing data for the particular step."""

from typing import Dict, Union, Optional, Any

import numpy as np

from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.log_events.step_event_log import StepEventLog


class StateRecord:
    """Keeps trajectory data for one step. Note: It should be ensured that the components are not going to
    change after assigning them to the step record (e.g. by copying the relevant ones, especially state and
    the serializable components).

    :param env_time: Internal time of environment (if available) that this record belongs to.
    :param maze_state: Current MazeState of the env.
    :param maze_action: Last MazeAction taken by the agent.
    :param step_event_log: Log of events dispatched by the env during the last step.
    :param reward: Reward as returned by the environment (either scalar or distributed reward)
    :param done: Dictionary indicating whether the environment or particular agents are done
    :param info: Dictionary with any other supplementary information provided by the env
    :param serializable_components: dict of all serializable components as provided by the env
        - e.g. { "demand_generator" : demand_generator_object }
    """

    def __init__(self,
                 env_time: Optional[int],
                 maze_state: MazeStateType,
                 maze_action: Optional[MazeActionType],
                 step_event_log: Optional[StepEventLog] = None,
                 reward: Optional[Union[float, np.ndarray, Any]] = None,
                 done: Optional[bool] = None,
                 info: Optional[Dict] = None,
                 serializable_components: Optional[Dict[str, Any]] = None):
        self.env_time = env_time
        self.maze_state = maze_state
        self.maze_action = maze_action
        self.step_event_log = step_event_log
        self.reward = reward
        self.done = done
        self.info = info
        self.serializable_components = serializable_components
