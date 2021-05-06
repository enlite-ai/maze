"""Contains a reward clipping wrapper."""
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.wrappers.wrapper import RewardWrapper


class RewardClippingWrapper(RewardWrapper[MazeEnv]):
    """Clips original step reward to range [min, max].

    :param env: The underlying environment.
    :param min_val: Minimum allowed reward value.
    :param max_val: Maximum allowed reward value.
    """

    def __init__(self, env: MazeEnv, min_val: float, max_val: float):
        super().__init__(env)
        self.min_val = min_val
        self.max_val = max_val

    @override(RewardWrapper)
    def reward(self, reward: float) -> float:
        """Clips the original reward.

        :param reward: The original reward.
        :return: The clipped reward.
        """
        return min(max(self.min_val, reward), self.max_val)

    @override(SimulatedEnvMixin)
    def clone_from(self, env: 'RewardClippingWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self.env.clone_from(env)
