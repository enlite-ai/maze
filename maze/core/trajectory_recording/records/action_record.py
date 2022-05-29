"""Record of either agent actions or maze action objects."""
import os
import pickle
from collections import defaultdict
from typing import Dict

from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_action import MazeActionType
from maze.core.env.structured_env import ActorID


class ActionRecord:
    """Action record holding either maze_action objects of agent actions for later deterministic (seeded) replays.

    :param seed: The seed identifying the respective deterministic episode for the action record.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.maze_actions: Dict[int, MazeActionType] = dict()
        self.agent_actions: Dict[int, Dict[ActorID, ActionType]] = defaultdict(dict)

        # corresponding reward for action record (useful for deterministic replay checks)
        self.cum_action_record_reward = None

    @classmethod
    def load(cls, dump_file: str) -> "ActionRecord":
        """Load existing action record from file.

        :param dump_file: Path to dumped action record.
        :return: Loaded action record.
        """
        assert os.path.exists(dump_file)
        with open(dump_file, 'rb') as fp:
            action_record = pickle.load(fp)
        return action_record

    def __len__(self) -> int:
        return max(len(self.maze_actions), len(self.agent_actions))

    def set_maze_action(self, step: int, maze_action: MazeActionType) -> None:
        """Add action item for specified step.

        :param step: The respective time step.
        :param maze_action: The maze action.
        """
        self.maze_actions[step] = maze_action

    def set_agent_action(self, step: int, actor_id: ActorID, action: ActionType) -> None:
        """Add action item for specified step.

        :param step: The respective time step.
        :param actor_id: ID of the actor to record the action for.
        :param action: The respective agent action.
        """
        self.agent_actions[step][actor_id] = action

    def get_maze_action(self, step: int) -> MazeActionType:
        """Request maze action item for specific step.

        :param step: The requested time step.
        :return: The maze action item.
        """
        return self.maze_actions[step]

    def get_agent_action(self, step: int, actor_id: ActorID) -> ActionType:
        """Request agent action item for specific step.

        :param step: The requested time step.
        :param actor_id: ID of the actor to query the action for.
        :return: The action item.
        """
        return self.agent_actions[step][actor_id]

    def dump(self, dump_file: str) -> None:
        """Dump action record to file.

        :param dump_file: absolute file path to pickle dump file.
        """
        with open(dump_file, 'wb') as fp:
            pickle.dump(self, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def clip(self, max_items: int) -> None:
        """Clip action record to maximum length.

        :param max_items: Maximum number of items in action record.
        """
        self.maze_actions = dict([(key, value) for key, value in
                                  list(self.maze_actions.items())[:max_items]])

        self.agent_actions = dict([(key, value) for key, value in
                                   list(self.agent_actions.items())[:max_items]])