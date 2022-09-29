"""Abstract class for rollout runners."""
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Any

import numpy as np
from omegaconf import DictConfig

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import StructuredEnv
from maze.core.utils.config_utils import EnvFactory, SwitchWorkingDirectoryToInput
from maze.core.utils.factory import Factory, ConfigType, CollectionOfConfigType
from maze.core.utils.seeding import MazeSeeding
from maze.core.wrappers.time_limit_wrapper import TimeLimitWrapper
from maze.core.wrappers.trajectory_recording_wrapper import TrajectoryRecordingWrapper
from maze.runner import Runner
from maze.utils.bcolors import BColors


class RolloutRunner(Runner, ABC):
    """General abstract class for rollout runners.

    Offers general structure, plus a couple of helper methods for env instantiation and performing the rollout.

    :param n_episodes: Count of episodes to run. If explicit seeds are given the actual number of episodes is given by
                       min(n_episodes, n_seeds).
    :param max_episode_steps: Count of steps to run in each episode (if environment returns done, the episode
                              will be finished earlier though).
    :param deterministic: Deterministic or stochastic action sampling.
    :param record_trajectory: Whether to record trajectory data.
    :param record_event_logs: Whether to record event logs.
    """

    def __init__(self,
                 n_episodes: int,
                 max_episode_steps: int,
                 deterministic: bool,
                 record_trajectory: bool,
                 record_event_logs: bool):
        self.n_episodes = n_episodes
        self.max_episode_steps = max_episode_steps
        self.deterministic = deterministic
        self.record_trajectory = record_trajectory
        self.record_event_logs = record_event_logs
        self._cfg: Optional[DictConfig] = None

        # keep track of the input directory
        self.input_dir = None

        # Generate a random state used for sampling random seeds for the envs and agents
        self.maze_seeding = MazeSeeding(env_seed=np.random.randint(np.iinfo(np.int32).max),
                                        agent_seed=np.random.randint(np.iinfo(np.int32).max),
                                        cudnn_determinism_flag=False, explicit_env_seeds=None,
                                        explicit_agent_seeds=None, shuffle_seeds=False)

    @override(Runner)
    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up prerequisites to rollouts.
        :param cfg: DictConfig defining components to initialize.
        """

        self._cfg = cfg
        self.input_dir = cfg.input_dir

        # Generate a random state used for sampling random seeds for the envs and agents
        self.maze_seeding = MazeSeeding(env_seed=cfg.seeding.env_base_seed, agent_seed=cfg.seeding.agent_base_seed,
                                        cudnn_determinism_flag=cfg.seeding.cudnn_determinism_flag,
                                        explicit_env_seeds=cfg.seeding.explicit_env_seeds,
                                        explicit_agent_seeds=cfg.seeding.explicit_agent_seeds,
                                        shuffle_seeds=cfg.seeding.shuffle_seeds)

    @override(Runner)
    def run(self) -> None:
        """Parse the supplied Hydra config and perform the run."""
        # If this is run from command line using Hydra, Hydra is by default configured to create
        # a fresh output directory for each run.
        # However, to ensure model states, normalization stats and else are loaded from expected
        # locations, we will change the dir back to the original working dir for the initialization
        # (and then change it back so that all later script output lands in the hydra output dir as expected)
        start_time = time.time()
        self.run_with(self._cfg.env, self._cfg.wrappers if "wrappers" in self._cfg else {}, self._cfg.policy)
        print(f'Rollout took {time.time() - start_time:.3f} seconds')

    @abstractmethod
    def run_with(self, env: ConfigType, wrappers: CollectionOfConfigType, agent: ConfigType) -> None:
        """Run the rollout with the given env, wrappers and agent configuration. A helper method to make rollouts
        easily runnable also directly from python, without building the hydra config object.

        Note that this method is designed to run only once -- if you call it from python directly (and not using
        Hydra from command line as is the main use case), you should respect this. Otherwise, you might get
        weird behavior especially from the statistics and events logging system, as the rollout runners register
        their own stats and event writers (so you might get duplicate stats) and order of operations sometimes
        matters (especially with parallel rollouts, where we do not want to carry the writers into child processes).

        :param env: Env config or object.
        :param wrappers: Wrappers config (see :class:`~maze.core.wrappers.wrapper_factory.WrapperFactory`).
        :param agent: Agent config or object.
        """

    @staticmethod
    def init_env_and_agent(env_config: DictConfig,
                           wrappers_config: CollectionOfConfigType,
                           max_episode_steps: int,
                           agent_config: DictConfig,
                           input_dir: str) -> (MazeEnv, Policy):
        """Build the environment (including wrappers) and agent according to given configuration.

        :param env_config: Environment config.
        :param wrappers_config: Wrapper config.
        :param max_episode_steps: Max number of steps per episode to limit the env for.
        :param agent_config: Policies config.
        :param input_dir: Directory to load the model from.

        :return: Tuple of (instantiated environment, instantiated agent).
        """

        with SwitchWorkingDirectoryToInput(input_dir):
            agent = Factory(base_type=Policy).instantiate(agent_config)

            env = EnvFactory(env_config, wrappers_config)()
            if not isinstance(env, TimeLimitWrapper):
                env = TimeLimitWrapper.wrap(env)
            env.set_max_episode_steps(max_episode_steps)

        return env, agent

    @staticmethod
    def run_episode(env: StructuredEnv, obs: ObservationType, agent: Policy,
                    deterministic: bool,
                    render: bool) -> None:
        """Helper function for running a single episode.

        :param env: Environment to run.
        :param obs: Initial observation, as returned by reset().
        :param deterministic: Argmax policy.
        :param agent: Agent to use.
        :param render: Whether to render the environment after every step.
        """

        done = False
        while not done:
            # inject the MazeEnv state if desired by the policy
            action = agent.compute_action(observation=obs,
                                          actor_id=env.actor_id(),
                                          maze_state=env.get_maze_state() if agent.needs_state() else None,
                                          env=env if agent.needs_env() else None,
                                          deterministic=deterministic)

            obs, rew, done, info = env.step(action)

            if render:
                assert isinstance(env, TrajectoryRecordingWrapper), "Rendering is supported only when " \
                                                                    "trajectory recording is enabled."
                env.render()

    @classmethod
    def run_interaction_loop(cls, env: StructuredEnv, agent: Policy, n_episodes: int,
                             env_seeds: List[Any], agent_seeds: List[Any], deterministic: bool,
                             render: bool = False, after_reset_callback: Callable = None) -> None:
        """Helper function for running the agent-environment interaction loop for specified number of steps
        and episodes.

        :param env: Environment to run.
        :param agent: Agent to use.
        :param n_episodes: Count of episodes to perform.
        :param env_seeds: The env seeds to be used for each episode.
        :param agent_seeds: The agent seeds to be used for each episode.
        :param render: Whether to render the environment after every step.
        :param after_reset_callback: If supplied, this will be executed after each episode to notify the observer.
        """

        for idx in range(n_episodes):
            env_seed, agent_seed = env_seeds[idx], agent_seeds[idx]
            env.seed(env_seed)
            agent.seed(agent_seed)
            try:
                obs = env.reset()
                agent.reset()
            except Exception as exception:
                BColors.print_colored(f'A error was encountered during reset on the env_seed: {env_seed} with '
                                      f'agent_seed: {agent_seed}', BColors.FAIL)
                raise exception

            if idx > 0:
                after_reset_callback()

            try:
                cls.run_episode(env=env, obs=obs, agent=agent,
                                deterministic=deterministic, render=render)
            except Exception as exception:
                BColors.print_colored(f'A error was encountered during rollout on the env_seed: {env_seed} with '
                                      f'agent_seed: {agent_seed}', BColors.FAIL)
                raise exception

        # Reset env and agent at the very end in order to collect the statistics
        env.reset()
        agent.reset()
        if after_reset_callback is not None:
            after_reset_callback()
