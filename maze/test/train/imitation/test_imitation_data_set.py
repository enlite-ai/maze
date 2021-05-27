import pickle
from typing import Any, Dict, Union, Tuple, List, Optional

import gym

from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.structured_env import ActorID
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.trajectory_recording.records.spaces_record import SpacesRecord
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.records.state_record import StateRecord
from maze.core.trajectory_recording.records.trajectory_record import StateTrajectoryRecord, SpacesTrajectoryRecord
from maze.core.wrappers.maze_gym_env_wrapper import make_gym_maze_env
from maze.core.wrappers.wrapper import ObservationWrapper
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.double import DoubleActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.double import \
    DoubleObservationConversion
from maze.test.shared_test_utils.run_maze_utils import run_maze_job
from maze.core.trajectory_recording.datasets.in_memory_dataset import InMemoryDataset


class _MockObservationStackWrapper(ObservationWrapper):
    """Example stateful wrapper -- observation depends on the observation from the previous step (both are stacked)."""

    def __init__(self, env: StructuredEnvSpacesMixin):
        super().__init__(env)
        self.last_observation_value = None

    def observation(self, observation: Any) -> Any:
        """Stacks observation with the last one."""
        assert list(observation.keys()) == ["observation"]
        observation_value = observation["observation"]
        stacked_observation = {"observation": [self.last_observation_value, observation_value]}
        self.last_observation_value = observation_value
        return stacked_observation

    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """If this is the first step in an episode, reset the observation stack."""
        if first_step_in_episode:
            self.last_observation_value = None

        return super().get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)


def _mock_state_trajectory_record(step_count: int):
    """Produce an episode record with maze_states and maze_actions corresponding to the step no."""
    episode_record = StateTrajectoryRecord("test")

    for i in range(step_count):
        episode_record.step_records.append(StateRecord(
            maze_state=i,
            maze_action=i if i < step_count - 1 else None,  # maze_action is not available in the last step
            step_event_log=StepEventLog(i),
            reward=0,
            done=i == step_count - 1,
            info=None,
            serializable_components={}
        ))

    return episode_record


def _mock_spaces_trajectory_record(step_count: int):
    """Produce an episode record with maze_states and maze_actions corresponding to the step no."""
    episode_record = SpacesTrajectoryRecord("test")

    for i in range(step_count):
        substep_record = SpacesRecord(
            actor_id=ActorID(0, 0),
            observation=dict(observation=i),
            action=dict(action=i),
            reward=0,
            done=i == step_count - 1
        )
        episode_record.step_records.append(StructuredSpacesRecord(substep_records=[substep_record]))

    return episode_record


def _env_factory():
    return DummyEnvironment(
        core_env=DummyCoreEnvironment(gym.spaces.Discrete(10)),
        action_conversion=[DoubleActionConversion()],
        observation_conversion=[DoubleObservationConversion()])


def test_state_record_load():
    dataset = InMemoryDataset(n_workers=1, conversion_env_factory=_env_factory)
    step_records = dataset.convert_trajectory(_mock_spaces_trajectory_record(5), dataset.conversion_env)

    # All steps should be loaded
    assert len(step_records) == 5

    # The original values should be kept (no conversion should take place)
    expected = [0, 1, 2, 3, 4]

    # Wrapping in the structured dict spaces
    expected_structured_actions = list(map(lambda x: {0: {"action": x}}, expected))
    expected_structured_observations = list(map(lambda x: {0: {"observation": x}}, expected))

    assert [rec.actions_dict for rec in step_records] == expected_structured_actions
    assert [rec.observations_dict for rec in step_records] == expected_structured_observations


def test_spaces_record_load():
    dataset = InMemoryDataset(n_workers=1, conversion_env_factory=_env_factory)
    step_records = dataset.convert_trajectory(_mock_state_trajectory_record(5), dataset.conversion_env)

    # Last step should be skipped, as no maze_action is available
    assert len(step_records) == 4

    # Both should be doubled by the dummy action interfaces from the original values of [0, 1, 2, 3]
    expected = [0, 2, 4, 6]

    # Wrapping in the structured dict spaces
    expected_structured_actions = list(map(lambda x: {0: {"action": x}}, expected))
    expected_structured_observations = list(map(lambda x: {0: {"observation": x}}, expected))

    assert [rec.actions_dict for rec in step_records] == expected_structured_actions
    assert [rec.observations_dict for rec in step_records] == expected_structured_observations


def test_data_load_with_stateful_wrapper():
    dataset = InMemoryDataset(n_workers=1, conversion_env_factory=lambda: _MockObservationStackWrapper.wrap(_env_factory()))
    step_records = dataset.convert_trajectory(_mock_state_trajectory_record(4), dataset.conversion_env)

    expected_observations = [
        {0: {"observation": [None, 0]}},
        {0: {"observation": [0, 2]}},
        {0: {"observation": [2, 4]}}
    ]
    assert [rec.observations_dict for rec in step_records] == expected_observations


def test_data_split():
    def _extract_observation_values_from(imitation_samples: List[Tuple[Dict, Dict]]):
        """Extract observation values from array of imitation samples of (obs, act) tuples"""
        return list(map(lambda sample: sample[0][0]["observation"], imitation_samples))

    dataset = InMemoryDataset(n_workers=1, conversion_env_factory=_env_factory)

    # Fill dataset with two episodes with 5 usable steps each
    for _ in range(2):
        dataset.append(_mock_state_trajectory_record(6))

    assert dataset.trajectory_references[0] == range(0, 5)
    assert dataset.trajectory_references[1] == range(5, 10)

    train, valid = dataset.random_split([5, 5])
    assert len(train) == 5
    assert len(valid) == 5
    assert _extract_observation_values_from(train) == [0, 2, 4, 6, 8]
    assert _extract_observation_values_from(valid) == [0, 2, 4, 6, 8]

    # Add two more episodes with 5 usable steps each (now we have 4 in total)
    for _ in range(2):
        dataset.append(_mock_state_trajectory_record(6))

    # 50:50 split
    train, valid = dataset.random_split([10, 10])
    assert len(train) == 10
    assert len(valid) == 10
    assert _extract_observation_values_from(train) == [0, 2, 4, 6, 8] * 2
    assert _extract_observation_values_from(valid) == [0, 2, 4, 6, 8] * 2

    # 75:25 split
    train, valid = dataset.random_split([15, 5])
    assert len(train) == 15
    assert len(valid) == 5
    assert _extract_observation_values_from(train) == [0, 2, 4, 6, 8] * 3
    assert _extract_observation_values_from(valid) == [0, 2, 4, 6, 8]

    # 80:20 split: Not satisfiable with episodes of 4 * 5 steps, should fall back on 15:5 ratio
    train, valid = dataset.random_split([16, 4])
    assert len(train) == 15
    assert len(valid) == 5
    assert _extract_observation_values_from(train) == [0, 2, 4, 6, 8] * 3
    assert _extract_observation_values_from(valid) == [0, 2, 4, 6, 8]

    # 70:30 split: Not satisfiable with episodes of 4 * 5 steps, should fall back on 10:10 ratio
    train, valid = dataset.random_split([14, 6])
    assert len(train) == 10
    assert len(valid) == 10
    assert _extract_observation_values_from(train) == [0, 2, 4, 6, 8] * 2
    assert _extract_observation_values_from(valid) == [0, 2, 4, 6, 8] * 2

    # 50:25:25 split
    train, test, valid = dataset.random_split([10, 5, 5])
    assert len(train) == 10
    assert len(test) == 5
    assert len(valid) == 5
    assert _extract_observation_values_from(train) == [0, 2, 4, 6, 8] * 2
    assert _extract_observation_values_from(test) == [0, 2, 4, 6, 8] * 1
    assert _extract_observation_values_from(valid) == [0, 2, 4, 6, 8] * 1


def test_parallel_data_load_from_directory():
    """Test loading trajectories of multiple episodes in parallel into an in-memory dataset. (Each
    data-loader process reads the files assigned to it.)"""
    # Heuristics rollout
    rollout_config = {
        "configuration": "test",
        "env": "gym_env",
        "env.name": "CartPole-v0",
        "policy": "random_policy",
        "runner": "sequential",
        "runner.n_episodes": 5,
        "runner.max_episode_steps": 3
    }
    run_maze_job(rollout_config, config_module="maze.conf", config_name="conf_rollout")

    dataset = InMemoryDataset(
        n_workers=2,
        conversion_env_factory=lambda: make_gym_maze_env("CartPole-v0"),
        dir_or_file="trajectory_data"
    )

    assert len(dataset) == 5 * 3


def test_parallel_data_load_from_file():
    trajectories = [_mock_spaces_trajectory_record(5)] * 10
    with open("trajectories.pkl", "wb") as out_ts:
        pickle.dump(trajectories, out_ts)

    dataset = InMemoryDataset(
        n_workers=2,
        conversion_env_factory=None,
        dir_or_file="trajectories.pkl"
    )

    assert len(dataset) == 5 * 10
