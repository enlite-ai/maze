"""Contains unit tests for default policies."""
from typing import Tuple, Sequence

import numpy as np

from maze.core.agent.default_policy import DefaultPolicy
from maze.core.agent.flat_policy import FlatPolicy
from maze.core.env.action_conversion import ActionType
from maze.core.env.observation_conversion import ObservationType


class DummyFlatPolicy(FlatPolicy):
    """ A dummy flat policy """

    def __init__(self, action_name):
        self.action_name = action_name

    def compute_action(self, observation: ObservationType, deterministic: bool) -> ActionType:
        """ compute action """
        return {self.action_name: np.ones(5, dtype=np.float32)}

    def compute_top_action_candidates(self, observation: ObservationType, num_candidates: int) -> \
            Tuple[Sequence[ActionType], Sequence[float]]:
        """ compute top action """
        raise NotImplementedError


def test_default_policy():
    """ unit tests """
    default_policy = DefaultPolicy({"policy_0": DummyFlatPolicy("action_0"),
                                              "policy_1": DummyFlatPolicy("action_1")})
    action = default_policy.compute_action(observation={}, policy_id="policy_0", deterministic=True)
    assert "action_0" in action
    action = default_policy.compute_action(observation={}, policy_id="policy_1", deterministic=True)
    assert "action_1" in action

    default_policy = DefaultPolicy({"policy_0": DummyFlatPolicy("action_0")})
    action = default_policy.compute_action(observation={}, policy_id=None, deterministic=True)
    assert "action_0" in action
