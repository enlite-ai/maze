"""
A more accessible Python API for training and rollout.
"""

import contextlib
import copy
import logging
import os
import sys
from typing import Callable, TypeVar, Union, Any, Dict, Optional, Mapping, List

import hydra.plugins.launcher
import omegaconf
from maze.api.config_auditor import ConfigurationAuditor
from maze.api.config_loader import ConfigurationLoader
from maze.api.utils import RunMode, InvalidSpecificationError, working_directory
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.log_stats.log_stats import LogStats, LogStatsLevel
from maze.core.rollout.rollout_runner import RolloutRunner
from maze.core.utils.factory import Factory
from maze.core.wrappers.wrapper import Wrapper
from maze.perception.models.critics import BaseStateCriticComposer
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.models.policies.base_policy_composer import BasePolicyComposer
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.trainers.common.config_classes import AlgorithmConfig
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.trainers.common.training_runner import TrainingRunner
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class RunContext:
    """
    RunContext offers convenient access to consistently configured training and rollout capabilities with minimal setup,
    yet is flexible enough to enable manipulation of every configurable aspect of Maze.
    It is initialized via an interface largely congruent with Maze' CLI, but also accepts instantiated Python objects.
    Internally it wraps a TrainingRunner and RolloutRunner object initiated w.r.t. to the specified configuration.

    Note: As of now, only training is supported. Rollout will be added soon.

    :param run_dir: Directory in which to store training and rollout processes and from which to read artefacts. This is
                    an alias of hydra.run.dir (i.e. "hydra.run.dir"=x in `overrides` has the same effect as run_dir=x).

    :param env: Environment configuration module name, Hydra configuration or callable returning instantiated Maze
                environment. It might be necessary, depending on the chosen runner and/or trainer configuration, that
                multiple environments have to be instantiated. env has to be passed as config, path to the config or
                factory function hence.

    :param wrappers: Wrapper configuration module name, Hydra configuration or instance.

    :param algorithm: Algorithm configuration module name, Hydra configuration or instance.

    :param model: Model configuration module name, Hydra configuration or instance.

    :param policy: Policy configuration module name, Hydra configuration or instance. Part of the model, i.e. setting
                   the policy via this argument or via model.policy in the overrides dictionary is equivalent.
                   Beware: When using a TemplateModelComposer the policy is to be specified without networks.

    :param critic: Critic configuration module name, Hydra configuration or instance. Part of the model, i.e. setting
                   the policy via this argument or via model.critic in the overrides dictionary is equivalent.
                   Beware: When using a TemplateModelComposer the critic is to be specified without networks.

    :param launcher: Launcher configuration module name, Hydra configuration or instance.

    :param runner: Runner configuration module name or Hydra configuration.
                   RolloutRunner configuration will be providable once rollouts are fully supported.

    :param overrides: Dictionary specifying overrides for individual properties. Overrides might specify values for
                      entire components like environments, some of their attributes or for specializations. Possible
                      values are Hydra configuration dictionaries as well as instantiated objects.
                      Beware that overrides will not load configuration modules and can only override loaded
                      configuration elements (i.e. overriding X.attr will fails if X is not part of the loaded
                      configuration).

    :param configuration: Determines which specialization configuration to load. Possible values: "run" or None. Has to
                          be specified via module name exclusively, i.e. configuration="test". This affects the
                          following components: Environments, models, algorithms and runners.

    :param experiment: Determines which experiment to load. Has to be specified via module name exclusively, i.e.
                       experiment="x".

    :param multirun: Allows running with multiple configurations (e.g. a grid search).

    :param silent: Whether to suppress output to stdout.

    """

    _SilenceReturnType = TypeVar("_SilenceReturnType")

    def __init__(
        self,
        # Auxiliary arguments providing explicit access to useful configuration properties.
        run_dir: Optional[str] = None,
        # Components.
        env: Optional[Union[str, Mapping[str, Any], Callable[[], MazeEnv]]] = None,
        wrappers: Optional[Union[str, Mapping[str, Any], Wrapper]] = None,
        algorithm: Optional[Union[str, Mapping[str, Any], AlgorithmConfig]] = None,
        model: Optional[Union[str, Mapping[str, Any], BaseModelComposer]] = None,
        policy: Optional[Union[str, Mapping[str, Any], BasePolicyComposer]] = None,
        critic: Optional[Union[str, Mapping[str, Any], BaseStateCriticComposer]] = None,
        launcher: Optional[Union[str, Mapping[str, Any], hydra.plugins.launcher.Launcher]] = None,
        runner: Optional[Union[str, Mapping[str, Any]]] = None,
        # Overrides.
        overrides: Optional[Dict[str, Union[Mapping[str, Any], Any]]] = None,
        # Configuration mode.
        configuration: Optional[str] = None,
        # Experiment module name.
        experiment: Optional[str] = None,
        # Whether to run in multirun mode.
        multirun: bool = False,
        # Whether to suppress output to stdout.
        silent: bool = False
    ):
        """
        The behaviour of this interface corresponds largely to the Maze CLI training API as documented in
        https://maze-rl.readthedocs.io/en/latest/trainers/maze_rllib_runner.html, i.e. defaults and functionality should
        be as similar as possible.
        https://maze-rl.readthedocs.io/en/latest/best_practices_and_tutorials/example_cmds.html lists examples using the
        CLI for training agents.

        Maze offers two different paradigms to instantiate objects: Directly via Python objects or by to utilizing Hydra
        to instantiate from configurations dictionaries. This API hence allows to pass (a) instantiated objects, (b)
        object configurations or (c) a reference determining which configuration to load.

        Major configurable components (environment, trainer, policy, ...) in the Maze framework are explicitly
        exposed here. These components can be naturally mapped to a single class.
        Components can be specified via
            1. the module name in the corresponding configuration group in maze.conf (e.g. env="cutting2d") to load.
            2. instantiated Python objects (e.g. env=Cutting2DEnvironment(...)).
            3. complete Hydra configuration dictionary for this component (e.g. env={"_target_": ...}).

        There are two exceptions to this: Environments and runners. Both are generated dynamically inside RunContext and
        hence cannot be passed as instantiated objects - alternatively, a factory method generating these objects can be
        provided.

        Beyond components, arbitrary (including nested) properties can be set using the overrides dictionary. This can
        be achieved by adding an entry to the `overrides` dictionary with {component_name.property_name: value}, e.g.:
        ```
        overrides = {
            "algorithm.eval_repeats": 5,
            "policy.device": "cpu",
            "algorithm_configuration": "dev"
        }
        ```
        Be aware that configuration file paths are not resolved, i.e. the specified value must be either a configuration
        dictionary or a primitive value.

        Finally, the configuration mode represents a special case. It allows to specify a particular set of
        configuration files that modify existing component configurations. Since the configuration mode modifies
        existing components instead of representing one itself, it can only be specified via its config module name
        (e.g. "test").

        Further remarks:
            - Every configuration option could be described in `overrides` without using the exposed arguments
              explicitly. In this case the initialization from module names is not possible in this case however.
            - The CLI is set up to execute either training or rollout, not both, as RunContext is. Hence RunContext
              accepts the arguments "training_runner" and "rollout_runner" instead of just one "runner".
        """

        if isinstance(env, MazeEnv):
            raise InvalidSpecificationError(
                "Environment must not be provided in instantiated form. Pass a factory function, Hydra configuration "
                "or reference therefore."
            )

        # Back up argv for restoration after __init__(). This is due to @hydra.main requiring necessary information
        # being specified via command line, which we emulate here.
        argv = copy.deepcopy(sys.argv)

        # Prepare and audit arguments.
        args = copy.copy(locals())

        self._auditors = {
            run_mode: ConfigurationAuditor(run_mode, args) for run_mode in (RunMode.TRAINING,)
        }
        for auditor in self._auditors.values():
            auditor.audit()

        # Prepare variables for ingestion by Hydra.
        self._configs: Dict[RunMode, List[DictConfig]] = {RunMode.TRAINING: [], RunMode.ROLLOUT: []}
        self._workdirs: List[str] = []
        self._silent = silent
        self._runners: Dict[RunMode, List[Union[TrainingRunner, RolloutRunner]]] = {
            RunMode.TRAINING: [], RunMode.ROLLOUT: []
        }

        # Restore original CLI arguments and working directory.
        sys.argv = argv

    def _generate_runners(self, run_mode: RunMode) -> List[TrainingRunner]:
        """
        Generates training or rollout runner(s).
        :param run_mode: Run mode. See See :py:class:`~maze.maze.api.RunMode`.
        :return: Instantiated Runner instance.
        """

        cl = ConfigurationLoader(
            _run_mode=run_mode,
            _kwargs=self._auditors[run_mode].kwargs,
            _overrides=self._auditors[run_mode].overrides,
            _ephemeral_init_kwargs=self._auditors[run_mode].ephemeral_init_kwargs
        )
        cl.load()

        self._workdirs = cl.workdirs
        self._configs[run_mode] = cl.configs
        runners: List[TrainingRunner] = []

        # Change to correct working directory (necessary due to being outside of Hydra scope).
        for workdir, config in zip(self._workdirs, self._configs[run_mode]):
            with working_directory(workdir):
                # Allow non-primitives in Hydra config.
                with omegaconf.flag_override(config, "allow_objects", True) as cfg:
                    # Set up and return runner.
                    runner = Factory(
                        base_type=TrainingRunner if run_mode == RunMode.TRAINING else RolloutRunner
                    ).instantiate(cfg.runner)
                    runner.setup(cfg)
                    runners.append(runner)
    
        return runners

    def train(self, n_epochs: Optional[int] = None, **train_kwargs) -> None:
        """
        Trains for specified number of epochs.
        After training the trainer is reset to the overall best state encountered.

        :param n_epochs: Number of epochs to train for.

        :param train_kwargs: Arguments to pass on to
                             :py:meth:`~maze.train.trainers.common.training_runner.TrainingRunner.run`.
        """

        if len(self._runners[RunMode.TRAINING]) == 0:
            self._runners[RunMode.TRAINING] = self._silence(lambda: self._generate_runners(RunMode.TRAINING))

        for i_runner, (workdir, runner) in enumerate(zip(self._workdirs, self._runners[RunMode.TRAINING])):
            with working_directory(workdir):
                self._silence(lambda: runner.run(n_epochs=n_epochs, **train_kwargs))

                # reset the runner and its policy to their overall best state if already dumped
                if os.path.exists(runner.state_dict_dump_file):
                    runner.trainer.load_state(runner.state_dict_dump_file)
                    # cope for --multirun setting
                    policy = self.policy[i_runner] if len(self._runners[RunMode.TRAINING]) > 1 else self.policy
                    policy.load_state_dict(state_dict=runner.trainer.state_dict())

    # To be updated after restructuring of (Rollout) runners.
    # def rollout(
    #     self,
    #     n_episodes: Optional[int] = None,
    #     max_episode_steps: Optional[int] = None,
    #     record_trajectory: Optional[bool] = None,
    #     record_event_logs: Optional[bool] = None,
    #     **rollout_kwargs
    # ) -> None:
    #     """
    #     Rolls out trainer's policy in specified environment.
    #     :param n_episodes: Count of episodes to run.
    #     :param max_episode_steps: Count of steps to run in each episode (if environment returns done, the episode
    #     will be finished earlier though).
    #     :param record_trajectory: Whether to record trajectory data.
    #     :param record_event_logs: Whether to record event logs.
    #     :param rollout_kwargs: Other arguments to pass on to rollout runner's __init__.
    #     """
    #
    #     assert "policy" not in rollout_kwargs, "Policy must be set at initialization time."
    #
    #     # Execute rollout.
    #     with working_directory(self._workdir):
    #         self._runners[RunMode.ROLLOUT] = self._silence(
    #             lambda: self._generate_runner(
    #                 RunMode.ROLLOUT,
    #                 kwargs=self._rollout_args,
    #                 overrides={
    #                     "hydra.run.dir": self._workdir,
    #                     **self._rollout_overrides,
    #                     **{
    #                         "runner." + key: val for key, val in locals().items()
    #                         if key not in ("self", "rollout_kwargs") and val is not None
    #                     },
    #                     **rollout_kwargs
    #                 }
    #             )
    #         )
    #
    #         self._silence(
    #             lambda: self._runners[RunMode.ROLLOUT].run_with(
    #                 env=self._runners[RunMode.TRAINING]._env_factory(),
    #                 wrappers=[],
    #                 agent=self._runners[RunMode.TRAINING]._model_composer.policy
    #             )
    #         )

    def _silence(self, task: Callable[[], _SilenceReturnType], dynamic: bool = True) -> _SilenceReturnType:
        """
        Suppresses output for execution of callable.
        :param task: Task to execute.
        :param dynamic: Whether to only silence a task if self._silent is True.
        """

        if dynamic and self._silent or not dynamic:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    return task()

        return task()

    @property
    def config(self) -> Dict[RunMode, Union[Optional[DictConfig], List[DictConfig]]]:
        """
        Returns Hydra DictConfigs specifying the configuration for training and rollout runners.

        :return: Dictionaries with DictConfig(s) for training and rollout each. Note that configurations are initialized
                 lazily, i.e. are not available until first training or rollout are initiated.

        """

        if len(self._workdirs) > 1:
            return self._configs

        return {key: value[0] if len(value) else None for key, value in self._configs}

    @property
    def run_dir(self) -> Union[Optional[str], List[str]]:
        """
        Returns run directory/directories.

        :return: Run directory/directories. Note that run directory are initialized lazily, i.e. are not available until
                 first training or rollout are initiated.
                 If run in single mode, list is collapsed to a single run directory string.

        """

        if len(self._workdirs) > 1:
            return self._workdirs

        return self._workdirs[0] if len(self._workdirs) else None

    @property
    def policy(self) -> Union[TorchPolicy, List[TorchPolicy]]:
        """
        Returns policy/policies.

        :return: Policy/policies used for training and rollout. If run in single mode, list is collapsed to a single
                 Policy instance.

        """

        policies = [runner.model_composer.policy for runner in self._runners[RunMode.TRAINING]]
        return policies if len(policies) > 1 else policies[0]

    def compute_action(
        self,
        observation: ObservationType,
        maze_state: Optional[MazeStateType] = None,
        env: Optional[BaseEnv] = None,
        actor_id: ActorID = None,
        deterministic: bool = False
    ) -> Union[ActionType, List[ActionType]]:
        """
        Computes action(s) with configured policy/policies.
        This wraps :meth:`maze.core.agent.policy.Policy.compute_action`.

        :return: Computed action(s) for next step. If run in single mode, list is collapsed to a single action instance.

        """

        actions = [
            runner.model_composer.policy.compute_action(observation, maze_state, env, actor_id, deterministic)
            for runner in self._runners[RunMode.TRAINING]
        ]

        return actions if len(actions) > 1 else actions[0]

    def evaluate(self, **eval_kwargs) -> Union[LogStats, List[LogStats]]:
        """
        Evaluates the trained/loaded policy with an RolloutEvaluator. By default 8 episodes are evaluated sequentially.

        :param eval_kwargs: kwargs to overwrite set (or default) initialization parameters for RolloutEvaluator. Note
                            that these arguments are ignored if RolloutRunner was passed as instance in AlgorithmConfig.

        :return: Logged statistics. One LogStats object if RunContext doesn't operate in multi-run mode, otherwise a
                 list thereof.

        """

        # Collect env factories and policies, wrap them in lists if they aren't already.
        env_factories = self.env_factory
        policies = self.policy
        if not isinstance(env_factories, List):
            env_factories = [env_factories]
            policies = [policies]

        # Generate rollout evaluators.
        rollout_evaluators: List[RolloutEvaluator] = []
        for runner, env_fn in zip(self._runners[RunMode.TRAINING], env_factories):
            # If rollout evaluator is not specified at all, create incomplete config with target.
            try:
                ro_eval = runner.cfg.algorithm.rollout_evaluator
            except omegaconf.errors.ConfigAttributeError:
                ro_eval = {"_target_": "maze.train.trainers.common.evaluators.rollout_evaluator.RolloutEvaluator"}

            # Override with specified arguments.
            if isinstance(ro_eval, DictConfig):
                ro_eval = omegaconf.OmegaConf.to_object(ro_eval)
            if isinstance(ro_eval, dict):
                ro_eval = {**ro_eval, **eval_kwargs}

            # Try to instantiate rollout runner directly from config. Works if completely specified in config or present
            # as instance of RolloutEvaluator.
            try:
                ro_eval = Factory(RolloutEvaluator).instantiate(ro_eval)
            # Merge with default values in case of incomplete RolloutEvaluator config.
            except TypeError:
                default_params = {
                    "eval_env": SequentialVectorEnv(env_factories=[env_fn]),
                    "n_episodes": 8,
                    "model_selection": None,
                    "deterministic": False
                }
                ro_eval = Factory(RolloutEvaluator).instantiate({**default_params, **ro_eval})
            finally:
                rollout_evaluators.append(ro_eval)
        # Evaluate policies.
        stats = [
            self._silence(
                lambda: [ro_eval.evaluate(policy), ro_eval.eval_env.get_stats(LogStatsLevel.EPOCH).last_stats][-1]
            )
            for env_factory, policy, ro_eval in zip(env_factories, policies, rollout_evaluators)
        ]

        return stats[0] if len(stats) == 0 else stats

    @property
    def env_factory(self) -> Union[Callable[[], MazeEnv], List[Callable[[], MazeEnv]]]:
        """
        Returns a newly generated environment with wrappers applied w.r.t. the specified configuration.

        :return: Environment factory function(s). One factory function if RunContext doesn't operate in multi-run mode,
                 otherwise a list thereof.

        """

        if len(self._runners[RunMode.TRAINING]) == 0:
            self._runners[RunMode.TRAINING] = self._silence(lambda: self._generate_runners(RunMode.TRAINING))

        runners = self._runners[RunMode.TRAINING]

        return runners[0].env_factory if len(runners) == 1 else [runner.env_factory for runner in runners]
