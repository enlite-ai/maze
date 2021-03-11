"""Tests for action and observation conversion for use cases such as imitation learning.

The provided MazeState and MazeAction should bubble from the MazeEnv up through the whole hierarchy.

Here, we test that all the base classes for wrappers (ActionWrapper, ObservationWrapper, RewardWrapper)
as well as the main monitoring wrappers (LogStatsWrapper, TrajectoryRecordingWrapper) handle the conversion
as expected.

MazeState and MazeAction are simulated by an integer working as a counter. In places where the MazeAction/MazeState
would be modified, the counter is incremented (like at space conversion interfaces or Action/Observation wrapper).
The rest of the wrappers (like the monitoring ones) should pass the MazeState and MazeAction counters further
unmodified.
"""

from typing import Any

import gym

from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.time_limit_wrapper import TimeLimitWrapper
from maze.core.wrappers.trajectory_recording_wrapper import TrajectoryRecordingWrapper
from maze.core.wrappers.wrapper import ObservationWrapper, ActionWrapper, RewardWrapper
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.double import DoubleActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.double import \
    DoubleObservationConversion


class _DummyObservationWrapper(ObservationWrapper):
    def observation(self, observation: dict) -> dict:
        """Increments the test counter by one"""
        observation["observation"] += 1
        return observation


class _DummyActionWrapper(ActionWrapper):
    def action(self, action: dict) -> dict:
        """Not used"""
        raise NotImplementedError

    def reverse_action(self, action: dict) -> dict:
        """Increments the test counter by one"""
        action["action"] += 1
        return action


class _DummyRewardWrapper(RewardWrapper):
    def reward(self, reward: Any) -> Any:
        """Not used"""
        raise NotImplementedError


def _build_env():
    env = DummyEnvironment(
        core_env=DummyCoreEnvironment(gym.spaces.Discrete(10)),
        action_conversion=[{"_target_": DoubleActionConversion}],
        observation_conversion=[{"_target_": DoubleObservationConversion}])

    env = _DummyActionWrapper.wrap(env)
    env = _DummyObservationWrapper.wrap(env)
    env = _DummyRewardWrapper.wrap(env)
    env = TimeLimitWrapper.wrap(env)
    env = LogStatsWrapper.wrap(env)
    env = TrajectoryRecordingWrapper.wrap(env)

    return env


def test_maze_state_and_action_conversion():
    env = _build_env()
    obs_dict, act_dict = env.get_observation_and_action_dicts(maze_state=1, maze_action=1, first_step_in_episode=True)

    # The expected output is
    # - Both the maze_state and maze_action should be put into a dictionary (key 0 indicating substep 0)
    # - Both should be equal to 3:
    #    - First, both are doubled the dummy space interfaces and wrapped in a dict
    #    - Second, both are incremented by the ActionWrapper, resp. ObservationWrapper
    #    - The remaining wrappers (like LogStats or TrajectoryRecording) should leave them as is
    assert act_dict == {0: {"action": 3}}
    assert obs_dict == {0: {"observation": 3}}


def test_observation_only_conversion():
    env = _build_env()
    obs_dict, act_dict = env.get_observation_and_action_dicts(maze_state=1, maze_action=None,
                                                              first_step_in_episode=True)

    # No wrapper in the env stack is multi-step => all of them should support state-only conversion.
    # The expected output of action dict should be the same as when converting both maze_state and maze_action
    # (see above).
    assert act_dict is None
    assert obs_dict == {0: {"observation": 3}}


def test_action_only_conversion():
    env = _build_env()
    obs_dict, act_dict = env.get_observation_and_action_dicts(maze_state=None, maze_action=1,
                                                              first_step_in_episode=True)

    # No wrapper in the env stack is multi-step => all of them should support maze_action-only conversion.
    # The expected output of observation dict should be the same as when converting both maze_state and maze_action
    # (see above)
    assert act_dict == {0: {"action": 3}}
    assert obs_dict is None
