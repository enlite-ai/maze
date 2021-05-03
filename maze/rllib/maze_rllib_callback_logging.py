""" Contains callbacks that enable maze logging capabilities. """
import os
import pickle
import pprint
from collections import defaultdict
from typing import Dict, Any

from ray.rllib.agents import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks

from maze.core.log_stats.log_stats import increment_log_step, LogStatsLevel, LogStatsAggregator, \
    register_log_stats_writer, get_stats_logger
from maze.core.log_stats.log_stats_writer_console import LogStatsWriterConsole
from maze.core.log_stats.log_stats_writer_tensorboard import LogStatsWriterTensorboard


class MazeRLlibLoggingCallbacks(DefaultCallbacks):
    """Callbacks to enable Maze-style logging."""

    def __init__(self):
        super().__init__()
        self.epoch_stats = None

    def init_logging(self, trainer_config: Dict[str, Any]) -> None:
        """Initialize logging.

        This needs to be done here as the on_train_result is not called in the main process, but in a worker process.

        Relies on the following:
          - There should be only one Callbacks object per worker process
          - The on train result should always be called in the same worker (if this is not the case,
            this system should still handle it, but it might mess things up)
        """
        assert self.epoch_stats is None, "Init logging should be called only once"

        # Local epoch stats -- stats from all envs will be collected here together
        self.epoch_stats = LogStatsAggregator(LogStatsLevel.EPOCH)
        self.epoch_stats.register_consumer(get_stats_logger("train"))

        # Initialize Tensorboard and console writers
        writer = LogStatsWriterTensorboard(log_dir='.', tensorboard_render_figure=True)
        register_log_stats_writer(writer)
        register_log_stats_writer(LogStatsWriterConsole())

        summary_writer = writer.summary_writer

        # Add config to tensorboard
        yaml_config = pprint.pformat(trainer_config)
        # prepare config text for tensorboard
        yaml_config = yaml_config.replace("\n", "</br>")
        yaml_config = yaml_config.replace(" ", "&nbsp;")
        summary_writer.add_text("job_config", yaml_config)

        # Load the figures from the given files and add them to tensorboard.
        network_files = filter(lambda x: x.endswith('.figure.pkl'), os.listdir('.'))
        for network_path in network_files:
            network_name = network_path.split('/')[-1].replace('.figure.pkl', '')
            fig = pickle.load(open(network_path, 'rb'))
            summary_writer.add_figure(f'{network_name}', fig, close=True)
            os.remove(network_path)

    def on_train_result(self, trainer: Trainer, result: dict, **kwargs) -> None:
        """Aggregates stats of all rollouts in one local aggregator and then writes them out.
        Called at the end of Trainable.train().

        :param trainer: Current model instance.
        :param result: Dict of results returned from model.train() call.
            You can mutate this object to add additional metrics.
        :param kwargs: Forward compatibility placeholder.
        """

        # Initialize the logging for this process if not done yet
        if self.epoch_stats is None:
            print("Initializing logging of train results")
            self.init_logging(trainer.config)

        # The main local aggregator should be empty
        #  - No stats should be collected here until we manually add them
        #  - Stats from the last call should be cleared out already (written out to the logs)
        assert self.epoch_stats.input == {}, "input should be empty at the beginning"

        # Get the epoch stats from the individual rollouts
        epoch_aggregators = trainer.workers.foreach_worker(
            lambda worker: worker.foreach_env(lambda env: env.get_stats(LogStatsLevel.EPOCH))
        )

        # Collect all episode stats from the epoch aggregators of individual rollout envs in the main local aggregator
        for worker_epoch_aggregator in epoch_aggregators:
            for env_epoch_aggregator in worker_epoch_aggregator:
                # Pass stats from the individual env runs into the main epoch aggregator
                for stats_key, stats_value in env_epoch_aggregator.input.items():
                    self.epoch_stats.input[stats_key].extend(stats_value)

        # clear logs at distributed workers
        def reset_episode_stats(env) -> None:
            """Empty inputs of the individual aggregators and make sure they don't have any consumers"""
            epoch_aggregator = env.get_stats(LogStatsLevel.EPOCH)
            epoch_aggregator.input = defaultdict(list)
            epoch_aggregator.consumers = []

        trainer.workers.foreach_worker(lambda worker: worker.foreach_env(lambda env: reset_episode_stats(env)))

        # Increment log step to trigger epoch logging
        increment_log_step()
