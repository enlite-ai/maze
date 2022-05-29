"""Contains an replay recorded actions policy."""
import os.path
from typing import Tuple
from typing import Union, Optional, Sequence

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.trajectory_recording.records.action_record import ActionRecord


class ReplayRecordedActionsPolicy(Policy):
    """A replay action record policy that executes in each (sub-)step the action stored in the provided action record.

    :param action_record_path: Action record or path to action record dump.
    :param with_agent_actions: If True agent actions are returned; else MazeActions.
    """

    def __init__(self, action_record_path: Optional[Union[ActionRecord, str]], with_agent_actions: bool):
        super().__init__()

        self._with_agent_actions = with_agent_actions

        self.action_record = None
        if action_record_path is not None:
            self.load_action_record(action_record_path)

    def load_action_record(self, action_record_path: Union[ActionRecord, str]) -> None:
        """Load action record from file.

        :param action_record_path: Action record or path to action record dump.
        """
        if isinstance(action_record_path, ActionRecord):
            self.action_record = action_record_path
        else:
            assert os.path.exists(action_record_path)
            self.action_record = ActionRecord.load(action_record_path)

    @override(Policy)
    def seed(self, seed: int) -> None:
        """Seed the policy."""

    @override(Policy)
    def needs_state(self) -> bool:
        """This policy requires the state object to compute the action."""
        return True

    @override(Policy)
    def needs_env(self) -> bool:
        """This policy does not require the env object to compute the action."""
        return True

    @override(Policy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[MazeStateType],
                       env: Optional[MazeEnv],
                       actor_id: Optional[ActorID] = None,
                       deterministic: bool = False) -> Union[ActionType, MazeActionType]:
        """Deterministically returns the action record action at the respective step.
        """
        current_env_time = env.get_env_time()

        if self._with_agent_actions:
            if actor_id is None:
                actor_id = env.actor_id()
            return self.action_record.get_agent_action(current_env_time, actor_id=actor_id)
        else:
            return self.action_record.get_maze_action(current_env_time)

    @override(Policy)
    def compute_top_action_candidates(self,
                                      observation: ObservationType,
                                      num_candidates: int,
                                      maze_state: Optional[MazeStateType],
                                      env: Optional[MazeEnv],
                                      actor_id: Union[str, int] = None) -> Tuple[Sequence[ActionType], Sequence[float]]:
        """
        Implementation of :py:attr:`~maze.core.agent.policy.Policy.compute_top_action_candidates`.
        """
        raise NotImplementedError
