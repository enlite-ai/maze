"""Runner interface for running Maze from CLI."""

from abc import ABC, abstractmethod

from omegaconf import DictConfig


class Runner(ABC):
    """Runner interface for running Maze from CLI.

    This class will be instantiated from the config obtained from hydra (cfg.runner). Then, the run method
    will be called, being supplied the whole hydra config (cfg).
    """

    @abstractmethod
    def run(self, cfg: DictConfig) -> None:
        """Perform the run.

        :param cfg: Config of the hydra job.
        """
