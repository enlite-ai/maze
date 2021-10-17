"""Allow getting multiple MazeAction candidates from the policy through the agent integration wrapper,
with all the MazeActions being passed through the whole wrapper stack."""

from typing import Sequence, Any, Tuple

from gym import spaces

from maze.core.env.action_conversion import ActionConversionInterface
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType


class ActionCandidates:
    """
    Action object for encapsulation of multiple action objects along with their respective probabilities.
    Useful when getting multiple candidate actions from a policy.

    :param candidates_and_probabilities: a tuple of sequences, where the first sequence corresponds to the possible
                                         actions, the other sequence to the associated probabilities
    """

    def __init__(self, candidates_and_probabilities: Tuple[Sequence[Any], Sequence[float]]):
        self.candidates = candidates_and_probabilities[0]
        self.probabilities = candidates_and_probabilities[1]
        assert len(self.candidates) == len(self.probabilities), \
            "candidates and probabilities should be of the same length"


class MazeActionCandidates:
    """
    MazeAction object for encapsulation of multiple MazeAction objects along with their respective probabilities.
    Useful when working with multiple candidate MazeActions from a policy.

    :param candidates: Candidate MazeActions
    :param probabilities: Respective probabilities
    """

    def __init__(self,
                 candidates: Sequence[MazeActionType],
                 probabilities: Sequence[float]):
        assert len(candidates) == len(probabilities), "candidates and probabilities should be of the same length"
        self.candidates = candidates
        self.probabilities = probabilities


class ActionConversionCandidatesInterface(ActionConversionInterface):
    """Wrapper for action conversion interface when working with multiple candidate actions/MazeActions.

    Wraps an action_conversion interface. When action is passed in, uses the wrapped interface to convert all
    action candidates to respective MazeActions separately.

    :param action_conversion: Underlying interface to apply to each candidate
    """

    def __init__(self, action_conversion: ActionConversionInterface):
        self.action_conversion = action_conversion

    def space_to_maze(self, action: ActionCandidates, maze_state: MazeStateType) -> MazeActionCandidates:
        """
        Convert an action candidates object (containing multiple candidate actions) into corresponding
        MazeAction candidates object.

        :param action: Action candidates object, encapsulating multiple actions.
        :param maze_state: Current state of the environment.
        :return: MazeAction candidate object, encapsulating multiple MazeActions.
        """
        assert isinstance(action, ActionCandidates), "action must be of type ActionCandidates when working with this" \
                                                     "wrapper"
        return MazeActionCandidates(
            candidates=list(map(
                lambda c: self.action_conversion.space_to_maze(c, maze_state),
                action.candidates)),
            probabilities=action.probabilities
        )

    def space(self) -> spaces.Space:
        """Return the space defined by the underlying action conversion interface."""
        return self.action_conversion.space()
