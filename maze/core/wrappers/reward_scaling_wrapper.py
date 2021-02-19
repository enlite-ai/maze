"""Contains a reward scaling wrapper."""
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.wrapper import RewardWrapper


class RewardScalingWrapper(RewardWrapper[MazeEnv]):
    """Scales original step reward by a multiplicative scaling factor.

    :param env: The underlying environment.
    :param scale: Multiplicative reward scaling factor.
    """

    def __init__(self, env: MazeEnv, scale: float):
        super().__init__(env)
        self.scale = scale

    @override(RewardWrapper)
    def reward(self, reward: float) -> float:
        """Scales the original reward.

        :param reward: The original reward.
        :return: The scaled reward.
        """
        return reward * self.scale
