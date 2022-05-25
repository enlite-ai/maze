from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.wrapper import Wrapper


class StepSkipInResetWrapper(Wrapper[MazeEnv]):
    """Mock wrapper that steps the env in the reset function (corresponds to step-skipping during env reset)"""

    def reset(self):
        """Step the env twice during the reset function"""
        obs = self.env.reset()
        for i in range(2):
            obs, _, _, _ = self.step(self.env.action_space.sample())
        return obs
