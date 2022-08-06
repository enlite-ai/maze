""" Contains a parallel rollout runner that allows to collect features by replaying pre-computed action records. """
import glob
import logging
import os
import traceback
from multiprocessing import Queue, Process
from typing import Iterable

from omegaconf import DictConfig

from maze.core.agent.replay_recorded_actions_policy import ReplayRecordedActionsPolicy
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.rollout.parallel_rollout_runner import ParallelRolloutRunner, ParallelRolloutWorker, ExceptionReport
from maze.core.rollout.rollout_runner import RolloutRunner
from maze.core.utils.factory import ConfigType, CollectionOfConfigType
from maze.core.wrappers.observation_normalization.observation_normalization_utils import obtain_normalization_statistics
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper
from maze.core.wrappers.spaces_recording_wrapper import SpacesRecordingWrapper
from maze.utils.bcolors import BColors

logger = logging.getLogger('ActionRecordWorker')
logger.setLevel(logging.INFO)


class ActionRecordWorker(ParallelRolloutWorker):
    """Class encapsulating functionality performed in worker processes."""

    @staticmethod
    def run(env_config: DictConfig,
            wrapper_config: DictConfig,
            agent_config: DictConfig,
            deterministic: bool,
            max_episode_steps: int,
            record_trajectory: bool,
            input_directory: str,
            reporting_queue: Queue,
            seeding_queue: Queue) -> None:
        """Build the environment and run the rollout for the specified number of episodes.

        :param env_config: Hydra configuration of the environment to instantiate.
        :param wrapper_config: Hydra configuration of environment wrappers.
        :param agent_config: Hydra configuration of agent's policies.
        :param deterministic: Deterministic or stochastic action sampling.
        :param max_episode_steps: Max number of steps per episode to perform
                                    (episode might end earlier if env returns done).
        :param record_trajectory: Whether to record trajectory data.
        :param input_directory: Directory to load the model from.
        :param reporting_queue: Queue for passing the stats and event logs back to the main process after each episode.
        :param seeding_queue: Queue for retrieving seeds.
        """
        env_seed, agent_seed = None, None
        try:
            env, agent = RolloutRunner.init_env_and_agent(env_config, wrapper_config, max_episode_steps,
                                                          agent_config, input_directory)
            assert isinstance(agent, ReplayRecordedActionsPolicy)

            # Set up the wrappers
            if not isinstance(env, SpacesRecordingWrapper):
                BColors.print_colored("Adding SpacesRecordingWrapper on top of wrapper stack!",
                                      color=BColors.WARNING)
                env = SpacesRecordingWrapper.wrap(env)

            env, episode_recorder = ParallelRolloutWorker._setup_monitoring(env, record_trajectory)

            first_episode = True
            while True:
                if seeding_queue.empty():
                    if first_episode:
                        break

                    # after we finished the last seed, we need to reset env and agent to collect the
                    # statistics of the last rollout
                    try:
                        env.reset()
                        agent.reset()
                    except Exception as e:
                        logger.warning(
                            f"\nException in event collection reset() encountered: {e}"
                            f"\n{traceback.format_exc()}")

                    reporting_queue.put(episode_recorder.get_last_episode_data())
                    break

                # initialize replay action policy
                action_record_path = seeding_queue.get()
                agent.load_action_record(action_record_path)

                # request env seed
                env_seed = agent.action_record.seed
                env.seed(env_seed)

                try:
                    obs = env.reset()
                    agent.reset()

                    RolloutRunner.run_episode(
                        env=env, agent=agent, obs=obs, deterministic=deterministic, render=False)

                    out_txt = f"agent_seed: {agent_seed}" \
                              f" | {str(env.core_env if isinstance(env, MazeEnv) else env)}"
                    logger.info(out_txt)
                except Exception as e:
                    out_txt = f"agent_seed: {agent_seed}" \
                              f" | {str(env.core_env if isinstance(env, MazeEnv) else env)}" \
                              f"\nException encountered: {e}" \
                              f"\n{traceback.format_exc()}"
                    logger.warning(out_txt)
                finally:
                    if not first_episode:
                        reporting_queue.put(episode_recorder.get_last_episode_data())
                    first_episode = False

        except Exception as exception:
            # Ship exception along with a traceback to the main process
            exception_report = ExceptionReport(exception, traceback.format_exc(), env_seed, agent_seed)
            reporting_queue.put(exception_report)
            raise


class ActionRecordRolloutRunner(ParallelRolloutRunner):
    """Parallel rollout runner that allows to collect features by replaying pre-computed action records.

    :param max_episode_steps: The maximum number of agent actions to take (careful these are not internal env steps).
    :param deterministic: Deterministic or stochastic action sampling.
    :param action_record_path: Path to action records.
    :param normalization_samples: Number of samples (=steps) to collect normalization statistics.
    :param n_processes: Count of processes to spread the rollout across.
    :param verbose: If True debug messages are printed to the command line.
    """

    def __init__(self,
                 max_episode_steps: int,
                 deterministic: bool,
                 action_record_path: str,
                 normalization_samples: int,
                 n_processes: int,
                 verbose: bool):
        super().__init__(n_episodes=0, max_episode_steps=max_episode_steps, deterministic=deterministic,
                         record_trajectory=False, record_event_logs=False, n_processes=n_processes)
        self.verbose = verbose

        self.action_record_paths = glob.glob(os.path.join(action_record_path, "*.pkl"))
        self.n_episodes = len(self.action_record_paths)
        self.normalization_samples = normalization_samples

        self.deterministic = True

    @override(ParallelRolloutRunner)
    def run_with(self, env: ConfigType, wrappers: CollectionOfConfigType, agent: ConfigType):
        """Run the parallel rollout in multiple worker processes."""
        # initialize observation normalization
        obs_norm_env, agent = ParallelRolloutRunner.init_env_and_agent(env_config=env, wrappers_config=wrappers,
                                                                       max_episode_steps=self.max_episode_steps,
                                                                       agent_config=agent,
                                                                       input_dir=self.input_dir)

        if isinstance(obs_norm_env, ObservationNormalizationWrapper):
            normalization_statistics = obtain_normalization_statistics(obs_norm_env,
                                                                       n_samples=self.normalization_samples)
            obs_norm_env.set_normalization_statistics(normalization_statistics)

        super().run_with(env=env, wrappers=wrappers, agent=agent)

    @override(ParallelRolloutRunner)
    def _launch_workers(self, env: ConfigType, wrappers: CollectionOfConfigType, agent: ConfigType) \
            -> Iterable[Process]:
        """Configure the workers according to the rollout config and launch them."""

        self.seeding_queue = Queue()
        for path in self.action_record_paths:
            self.seeding_queue.put(path)

        n_plans = len(self.action_record_paths)
        actual_number_of_episodes = min(n_plans, self.n_episodes)
        if actual_number_of_episodes < self.n_episodes:
            BColors.print_colored(f'Only {n_plans} explicit seed(s) given, thus the number of episodes changed '
                                  f'from: {self.n_episodes} to {actual_number_of_episodes}.', BColors.WARNING)

        # Configure and launch the processes
        workers = self._configure_and_launch_processes(parallel_worker_type=ActionRecordWorker,
                                                       env=env, wrappers=wrappers, agent=agent)
        return workers
