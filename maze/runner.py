"""Runner interface for running Maze from CLI."""

from abc import ABC, abstractmethod

from omegaconf import DictConfig

from maze.core.utils.seeding import MazeSeeding


class Runner(ABC):
    """Runner interface for running Maze from CLI.

    This class will be instantiated from the config obtained from hydra (cfg.runner). Then, the run method
    will be called, being supplied the whole hydra config (cfg).
    """
    maze_seeding: MazeSeeding

    @abstractmethod
    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up runner.
        :param cfg: Config of the hydra job.
        """

    @abstractmethod
    def run(self, **kwargs) -> None:
        """Perform the run.
        :param kwargs: Runner-specific arguments.
        """
