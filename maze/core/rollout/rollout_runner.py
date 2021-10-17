"""Abstract class for rollout runners."""
from abc import ABC, abstractmethod
from typing import Callable, Optional

from omegaconf import DictConfig

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.base_env import BaseEnv
from maze.core.env.structured_env import StructuredEnv
from maze.core.utils.config_utils import EnvFactory, SwitchWorkingDirectoryToInput
from maze.core.utils.factory import Factory, ConfigType, CollectionOfConfigType
from maze.core.utils.seeding import MazeSeeding
from maze.core.wrappers.time_limit_wrapper import TimeLimitWrapper
from maze.core.wrappers.trajectory_recording_wrapper import TrajectoryRecordingWrapper
from maze.runner import Runner


class RolloutRunner(Runner, ABC):
    """General abstract class for rollout runners.

    Offers general structure, plus a couple of helper methods for env instantiation and performing the rollout.

    :param n_episodes: Count of episodes to run.
    :param max_episode_steps: Count of steps to run in each episode (if environment returns done, the episode
                                will be finished earlier though).
    :param record_trajectory: Whether to record trajectory data.
    :param record_event_logs: Whether to record event logs.
    """

    def __init__(self,
                 n_episodes: int,
                 max_episode_steps: int,
                 record_trajectory: bool,
                 record_event_logs: bool):
        self.n_episodes = n_episodes
        self.max_episode_steps = max_episode_steps
        self.record_trajectory = record_trajectory
        self.record_event_logs = record_event_logs
        self._cfg: Optional[DictConfig] = None

        # keep track of the input directory
        self.input_dir = None

    @override(Runner)
    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up prerequisites to rollouts.
        :param cfg: DictConfig defining components to initialize.
        """

        self._cfg = cfg
        self.input_dir = cfg.input_dir

        # Generate a random state used for sampling random seeds for the envs and agents
        self.maze_seeding = MazeSeeding(cfg.seeding.env_base_seed, cfg.seeding.agent_base_seed,
                                        cfg.seeding.cudnn_determinism_flag)

    @override(Runner)
    def run(self) -> None:
        """Parse the supplied Hydra config and perform the run."""
        # If this is run from command line using Hydra, Hydra is by default configured to create
        # a fresh output directory for each run.
        # However, to ensure model states, normalization stats and else are loaded from expected
        # locations, we will change the dir back to the original working dir for the initialization
        # (and then change it back so that all later script output lands in the hydra output dir as expected)

        self.run_with(self._cfg.env, self._cfg.wrappers if "wrappers" in self._cfg else {}, self._cfg.policy)

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
                           input_dir: str,
                           env_instance_seed: int,
                           agent_instance_seed: int) -> (BaseEnv, Policy):
        """Build the environment (including wrappers) and agent according to given configuration.

        :param env_config: Environment config.
        :param wrappers_config: Wrapper config.
        :param max_episode_steps: Max number of steps per episode to limit the env for.
        :param agent_config: Policies config.
        :param input_dir: Directory to load the model from.
        :param env_instance_seed: The seed for this particular env.
        :param agent_instance_seed: The seed for this particular agent.

        :return: Tuple of (instantiated environment, instantiated agent).
        """

        with SwitchWorkingDirectoryToInput(input_dir):
            env = EnvFactory(env_config, wrappers_config)()
            if not isinstance(env, TimeLimitWrapper):
                env = TimeLimitWrapper.wrap(env)
            env.set_max_episode_steps(max_episode_steps)
            env.seed(env_instance_seed)

            agent = Factory(base_type=Policy).instantiate(agent_config)
            agent.seed(agent_instance_seed)

        return env, agent

    @staticmethod
    def run_interaction_loop(env: StructuredEnv, agent: Policy, n_episodes: int,
                             render: bool = False, episode_end_callback: Callable = None) -> None:
        """Helper function for running the agent-environment interaction loop for specified number of steps
        and episodes.

        :param env: Environment to run.
        :param agent: Agent to use.
        :param n_episodes: Count of episodes to perform.
        :param render: Whether to render the environment after every step.
        :param episode_end_callback: If supplied, this will be executed after each episode to notify the observer.
        """
        obs = env.reset()

        for _ in range(n_episodes):
            done = False
            while not done:
                # inject the MazeEnv state if desired by the policy
                action = agent.compute_action(observation=obs,
                                              actor_id=env.actor_id(),
                                              maze_state=env.get_maze_state() if agent.needs_state() else None,
                                              env=env if agent.needs_env() else None)

                obs, rew, done, info = env.step(action)

                if render:
                    assert isinstance(env, TrajectoryRecordingWrapper), "Rendering is supported only when " \
                                                                        "trajectory recording is enabled."
                    env.render()

            obs = env.reset()
            if episode_end_callback is not None:
                episode_end_callback()
