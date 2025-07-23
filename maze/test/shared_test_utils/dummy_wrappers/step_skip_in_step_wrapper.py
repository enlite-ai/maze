"""File holding wrappers to mimic (sub) step skipping."""
from __future__ import annotations

from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.log_events.skipping_events import SkipEvent
from maze.core.wrappers.wrapper import EnvType, Wrapper


class StepSkipInStepWrapper(Wrapper[MazeEnv]):
    """Mock wrapper that steps the env two times in the step function (corresponds to step-skipping)"""

    def step(self, action):
        """Step the env twice during the reset function"""
        self.env.step(action)
        return self.env.step(self.env.action_space.sample())


class ConfigurableStepSkipInStepWrapper(Wrapper[MazeEnv]):
    """Mock wrapper that steps the env multiple (configurable) times in the step method
    (corresponds to step-skipping).

    :param env: Environment to wrap.
    :param skip_sequence: Sequence of sub steps. False indicates no skipping, True indicates skipping.
    :param n_agents: Number of agents.
    """

    def __init__(self, env: EnvType, skip_sequence: list[bool], n_agents: int):
        super().__init__(env)

        self.skip_sequence = skip_sequence
        self.n_sub_steps = len(skip_sequence)
        self.cur_idx = 0

        self.n_agents = n_agents

        self._step_events = self.env.context.event_service.create_event_topic(SkipEvent)

    def step(self, action: ActionType):
        """Step the environment.

        :param action: The action to take. Not used.
        """
        step_val = None

        # Step for all agents
        self._step_events.sub_step(sub_step_is_skipped=False, substep_id=self.cur_idx)
        for _ in range(self.n_agents):
            step_val = self.env.step(self.env.action_space.sample())

        # No sub steps
        if self.n_sub_steps == 0:
            return step_val

        # Sub steps
        self.cur_idx += 1

        # Iterate the skipp-able consecutive sub steps
        while self.cur_idx < self.n_sub_steps and self.skip_sequence[self.cur_idx]:
            self._step_events.sub_step(sub_step_is_skipped=True, substep_id=self.cur_idx)
            for _ in range(self.n_agents):
                step_val = self.env.step(self.env.action_space.sample())

            self.cur_idx += 1

        # Check for flat step
        if self.cur_idx == self.n_sub_steps:
            # Mimic env done
            step_val = (step_val[0], step_val[1], True, step_val[3])

            # Check for flat step skip
            if sum(self.skip_sequence) == self.n_sub_steps:
                self._step_events.flat_step(flat_step_is_skipped=True)
            else:
                self._step_events.flat_step(flat_step_is_skipped=False)

        # Handle current index
        self.cur_idx %= self.n_sub_steps

        return step_val
