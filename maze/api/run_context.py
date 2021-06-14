"""
A more accessible Python API for training and rollout.
"""

import contextlib
import copy
import logging
import os
import sys
from typing import Callable, TypeVar, Union, Any, Dict, Optional, Mapping, Type

import hydra.plugins.launcher
import omegaconf
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv

from maze.train.trainers.common.model_selection.model_selection_base import ModelSelectionBase

from maze.train.parallelization.vector_env.subproc_vector_env import SubprocVectorEnv

from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator

from maze.train.trainers.common.evaluators.evaluator import Evaluator
from omegaconf import DictConfig

from maze.api.config_auditor import ConfigurationAuditor
from maze.api.config_loader import ConfigurationLoader
from maze.api.utils import RunMode, InvalidSpecificationError, working_directory, RunContextError
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.action_conversion import ActionType
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from maze.core.rollout.rollout_runner import RolloutRunner
from maze.core.utils.factory import Factory
from maze.core.wrappers.wrapper import Wrapper
from maze.perception.models.critics import BaseStateCriticComposer
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.models.policies.base_policy_composer import BasePolicyComposer
from maze.train.trainers.common.config_classes import AlgorithmConfig
from maze.train.trainers.common.training_runner import TrainingRunner

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

    :param runner: Runner configuration module name, Hydra configuration or instance. RolloutRunner configuration
                   will be providable once rollouts are fully supported.

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
        runner: Optional[Union[str, Mapping[str, Any], TrainingRunner]] = None,
        # Overrides.
        overrides: Optional[Dict[str, Union[Mapping[str, Any], Any]]] = None,
        # Configuration mode.
        configuration: Optional[str] = None,
        # Experiment module name.
        experiment: Optional[str] = None,
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
        args = copy.deepcopy(locals())
        self._auditors = {
            run_mode: ConfigurationAuditor(run_mode, args) for run_mode in (RunMode.TRAINING,)
        }
        for auditor in self._auditors.values():
            auditor.audit()

        # Prepare variables for ingestion by Hydra.
        self._configs: Dict[RunMode, Optional[DictConfig]] = {RunMode.TRAINING: None, RunMode.ROLLOUT: None}
        self._workdir: Optional[str] = None
        self._silent = silent
        self._runners: Dict[RunMode, Optional[Union[TrainingRunner, RolloutRunner]]] = {
            RunMode.TRAINING: None, RunMode.ROLLOUT: None
        }

        # Restore original CLI arguments and working directory.
        sys.argv = argv

    def _generate_runner(self, run_mode: RunMode) -> TrainingRunner:
        """
        Generates training or rollout runner.
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
        self._workdir = cl.workdir
        self._configs[run_mode] = cl.config

        # Change to correct working directory (necessary due to being outside of Hydra scope).
        with working_directory(self._workdir):
            # Allow non-primitives in Hydra config.
            with omegaconf.flag_override(self._configs[run_mode], "allow_objects", True) as cfg:
                # Set up and return runner.
                runner = Factory(
                    base_type=TrainingRunner if run_mode == RunMode.TRAINING else RolloutRunner
                ).instantiate(cfg.runner)
                runner.setup(cfg)
    
        return runner

    def train(self, n_epochs: Optional[int] = None, **train_kwargs) -> None:
        """
        Trains for specified number of epochs.

        :param n_epochs: Number of epochs to train for.

        :param train_kwargs: Arguments to pass on to
                             :py:meth:`~maze.train.trainers.common.training_runner.TrainingRunner.run`.
        """

        if self._runners[RunMode.TRAINING] is None:
            self._runners[RunMode.TRAINING]: TrainingRunner = self._silence(
                lambda: self._generate_runner(RunMode.TRAINING)
            )

        with working_directory(self._workdir):
            self._silence(lambda: self._runners[RunMode.TRAINING].run(n_epochs=n_epochs, **train_kwargs))

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
    def configs(self) -> Dict[RunMode, Optional[DictConfig]]:
        """
        Returns Hydra DictConfigs specifying the configuration for training and rollout runners.

        :return: Dictionary with one DictConfig for training and rollout each. Note that rollout config is None until
                 first rollout has been enacted.

        """

        return self._configs

    @property
    def run_dir(self) -> str:
        """
        Returns run directory.

        :return: Run directory.

        """

        return self._workdir

    @property
    def policy(self) -> TorchPolicy:
        """
        Returns policy.

        :return: Policy used for training and rollout.

        """

        return self._runners[RunMode.TRAINING].model_composer.policy

    def compute_action(
        self,
        observation: ObservationType,
        maze_state: Optional[MazeStateType] = None,
        env: Optional[BaseEnv] = None,
        actor_id: ActorID = None,
        deterministic: bool = False
    ) -> ActionType:
        """
        Computes action with configured policy. This wraps :meth:`maze.core.agent.policy.Policy.compute_action`.

        :return: Computed action for next step.

        """

        return self.policy.compute_action(observation, maze_state, env, actor_id, deterministic)

    def evaluate(
        self,
        n_envs: int,
        n_episodes: int,
        model_selection: Optional[ModelSelectionBase] = None,
        deterministic: bool = False,
        parallel: bool = False
    ) -> None:
        """
        Evaluates the trained/loaded policy with an RolloutEvaluator.

        :param n_envs: Number of environments.

        :param n_episodes: Number of evaluation episodes to run. Note that the actual number might be slightly larger
                           due to the distributed nature of the environment.

        :param model_selection: Model selection to notify about the recorded rewards.

        :param deterministic: Whether to compute the policy action deterministically.

        :param parallel: Whether to evaluate environments in parallel using SubprocVectorEnv.

        """

        # todo Would be better if we just load the policy instead of having to initialize a training runner (e.g. after
        #  training in a previous run and loading from its checkpoints).
        if self._runners[RunMode.TRAINING] is None:
            self._runners[RunMode.TRAINING]: TrainingRunner = self._silence(
                lambda: self._generate_runner(RunMode.TRAINING)
            )

        evaluator = RolloutEvaluator(
            eval_env=(SubprocVectorEnv if parallel else SequentialVectorEnv)(
                env_factories=[self.env_factory] * n_envs
            ),
            n_episodes=n_episodes,
            model_selection=model_selection,
            deterministic=deterministic
        )

        evaluator.evaluate(self.policy)

    @property
    def env_factory(self) -> Callable[[], MazeEnv]:
        """
        Returns a newly generated environment with wrappers applied w.r.t. the specified configuration.

        :return: Environment factory function.

        """

        if self._runners[RunMode.TRAINING]:
            return self._runners[RunMode.TRAINING].env_factory
        elif self._runners[RunMode.ROLLOUT]:
            return self._runners[RunMode.ROLLOUT].env_factory
        else:
            raise RunContextError(
                "Neither training nor rollout runner are instantiated. .env_factory() is not available."
            )
