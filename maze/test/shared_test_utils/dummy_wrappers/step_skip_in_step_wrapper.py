from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.wrapper import Wrapper


class StepSkipInStepWrapper(Wrapper[MazeEnv]):
    """Mock wrapper that steps the env two times in the step function (corresponds to step-skipping)"""

    def step(self, action):
        """Step the env twice during the reset function"""
        self.env.step(action)
        return self.env.step(self.env.action_space.sample())
