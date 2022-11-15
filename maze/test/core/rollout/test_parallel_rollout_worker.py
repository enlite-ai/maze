import queue
from collections import deque
from typing import Any, Type, Optional

from maze.core.agent.random_policy import RandomPolicy
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.rollout.parallel_rollout_runner import ParallelRolloutWorker, EpisodeStatsReport
from maze.test.shared_test_utils.dummy_env.dummy_core_env import DummyCoreEnvironment
from maze.test.shared_test_utils.dummy_env.dummy_maze_env import DummyEnvironment
from maze.test.shared_test_utils.dummy_env.space_interfaces.action_conversion.dict import DictActionConversion
from maze.test.shared_test_utils.dummy_env.space_interfaces.observation_conversion.dict import ObservationConversion


class MockQueue:
    """Mock queue, just keeping the items that are passed to it."""

    def __init__(self):
        self.items = deque()

    def put(self, item: Any) -> None:
        self.items.append(item)

    def get(self, block: Optional[bool] = None) -> Any:
        if block is False:
            if not len(self.items):
                raise queue.Empty

        return self.items.popleft()

    def empty(self) -> bool:
        return len(self.items) == 0


def build_test_maze_env(env_type: Type) -> DummyEnvironment:
    """Builds a dummy maze env with the given env_type (should be a subclass of DummyEnvironment)"""
    observation_conversion = ObservationConversion()

    return env_type(
        core_env=DummyCoreEnvironment(observation_conversion.space()),
        action_conversion=[DictActionConversion()],
        observation_conversion=[observation_conversion]
    )


class ErrorInResetEnv(DummyEnvironment):
    """Raises an error in every second reset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_episodes = 0

    def reset(self) -> ObservationType:
        if self.n_episodes % 2:
            raise RuntimeError("Test error in reset")
        self.n_episodes += 1
        return super().reset()


class ErrorInStepEnv(DummyEnvironment):
    """Raises an error in the step function of every second episode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_episodes = 0

    def reset(self) -> ObservationType:
        self.n_episodes += 1
        return super().reset()

    def step(self, *args, **kwargs):
        if self.n_episodes % 2:
            raise RuntimeError("Test error in step")
        return super().step(*args, **kwargs)


def _run_test_rollout(env: MazeEnv, n_episodes: int = 5):
    reporting_queue = MockQueue()
    seeding_queue = MockQueue()

    for _ in range(n_episodes):
        seeding_queue.put((1337, 1337))

    ParallelRolloutWorker.run(
        env_config=env,
        wrapper_config={},
        agent_config=RandomPolicy(action_spaces_dict=env.action_spaces_dict),
        deterministic=False,
        max_episode_steps=3,
        record_trajectory=False,
        input_directory=None,
        reporting_queue=reporting_queue,
        seeding_queue=seeding_queue
    )

    assert len(reporting_queue.items) == n_episodes

    for r in reporting_queue.items:
        assert isinstance(r, EpisodeStatsReport)


def test_handles_regular_rollout():
    env = build_test_maze_env(DummyEnvironment)
    _run_test_rollout(env)


def test_handles_rollout_with_errors_in_reset():
    env = build_test_maze_env(ErrorInResetEnv)
    _run_test_rollout(env)


def test_handles_rollout_with_errors_in_steps():
    env = build_test_maze_env(ErrorInStepEnv)
    _run_test_rollout(env)


def test_handles_rollout_with_error_in_last_reset():
    env = build_test_maze_env(ErrorInResetEnv)
    # The env raises an error in every second reset => with one episode, this will be the last reset
    # for stats collection
    _run_test_rollout(env, n_episodes=1)


def test_handles_empty_seeding_queue():
    env = build_test_maze_env(ErrorInResetEnv)
    _run_test_rollout(env, n_episodes=0)
