"""Custom Hydra launcher distributing the jobs in separate processes on the local machine."""
import logging
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional

from hydra import TaskFunction
from hydra.core.config_loader import ConfigLoader
from hydra.core.config_store import ConfigStore
from hydra.core.utils import (
    JobReturn,
    configure_log,
    filter_overrides,
    setup_globals, run_job,
)
from hydra.plugins.launcher import Launcher
from omegaconf import DictConfig, open_dict

from maze.utils.process import Process

logger = logging.getLogger(__name__)


@dataclass
class LauncherConfig:
    """Hardcoded launcher configuration, linking the hydra/launcher=local override to the MazeLocalLauncher class"""
    _target_: str = "hydra_plugins.maze_local_launcher.MazeLocalLauncher"

    # maximum number of concurrently running jobs. if -1, all CPUs are used
    n_jobs: int = -1


ConfigStore.instance().store(
    group="hydra/launcher", name="local", node=LauncherConfig
)


class MazeLocalLauncher(Launcher):
    """Custom Hydra launcher distributing the jobs in separate processes on the local machine.

    The implementation is based on
    https://github.com/facebookresearch/hydra/blob/master/examples/plugins/example_launcher_plugin/hydra_plugins/example_launcher_plugin/example_launcher.py.

    :param n_jobs: Maximum number of parallel jobs. If -1, all CPUs are used.
    """

    def __init__(self, n_jobs: int):
        self.config: Optional[DictConfig] = None
        self.config_loader: Optional[ConfigLoader] = None
        self.task_function: Optional[TaskFunction] = None

        # determine number of parallel jobs
        self._n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

    def setup(self, config: DictConfig, config_loader: ConfigLoader, task_function: TaskFunction) -> None:
        """Implementation of Launcher.setup, called before the launch

        :param config: The master config
        :param config_loader: The config loader, used to derive the job configurations from the sweep run.
        :param task_function: The job entry point as function object. This is not used at all, as it is much simpler
            to call the same command from bash than to serialize the function object, transfer it to the pod and
            calling the deserialized method there.
        """
        self.config = config
        self.config_loader = config_loader
        self.task_function = task_function

    def launch(self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int) -> Sequence[JobReturn]:
        """Implementation of Launcher.launch

        :param job_overrides: a List of List<String>, where each inner list is the arguments for one job run.
        :param initial_job_idx: Initial job idx in batch.
        :return: an array of return values from run_job with indexes corresponding to the input list indexes.
        """
        setup_globals()
        assert self.config is not None
        assert self.config_loader is not None
        assert self.task_function is not None

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = self.config.hydra.sweep.dir
        Path(str(sweep_dir)).mkdir(parents=True, exist_ok=True)

        logger.info(f"Local Launcher is launching {len(job_overrides)} jobs locally")
        logger.info(f"Launching jobs, sweep output dir : {sweep_dir}")
        for idx, overrides in enumerate(job_overrides):
            logger.info("\t#{} : {}".format(idx, " ".join(filter_overrides(overrides))))

        results = []
        workers = []
        for i, overrides in enumerate(job_overrides):
            idx = initial_job_idx + i
            lst = " ".join(filter_overrides(overrides))
            logger.info(f"\t#{idx} : {lst}")

            sweep_config = self.config_loader.load_sweep_config(
                self.config, list(overrides)
            )
            with open_dict(sweep_config):
                sweep_config.hydra.job.id = f"job_id_for_{idx}"
                sweep_config.hydra.job.num = idx

            p = Process(target=run_job,
                        kwargs=dict(config=sweep_config,
                                    task_function=self.task_function,
                                    job_dir_key="hydra.sweep.dir",
                                    job_subdir_key="hydra.sweep.subdir"))
            p.start()
            workers.append(p)

            # wait for current/last batch of workers
            if ((i + 1) % self._n_jobs == 0) or ((i + 1) == len(job_overrides)):
                for w in workers:
                    w.join()

                    # forward exceptions from the workers
                    if w.exception():
                        raise w.exception()

                # book keeping
                results.extend([p.result() for p in workers])
                workers = []

        assert len(results) == len(job_overrides)
        return results
