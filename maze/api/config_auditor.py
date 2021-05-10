"""
Parses arguments and checks for their internal consistency.
"""

import copy
import dataclasses
from typing import Dict, Any, Tuple, Union, Mapping, Set, Type, Optional

import hydra
import omegaconf

from maze.api.utils import RunMode, InvalidSpecificationError, _PrimitiveType, _OverridesType, _ATTRIBUTE_PROXIES
from maze.runner import Runner
from maze.train.trainers.a2c.a2c_algorithm_config import A2CAlgorithmConfig
from maze.train.trainers.a2c.a2c_trainer import A2C
from maze.train.trainers.common.actor_critic.actor_critic_runners import ACRunner, ACDevRunner, ACLocalRunner
from maze.train.trainers.common.config_classes import AlgorithmConfig
from maze.train.trainers.es.es_algorithm_config import ESAlgorithmConfig
from maze.train.trainers.es.es_runners import ESDevRunner
from maze.train.trainers.imitation.bc_algorithm_config import BCAlgorithmConfig
from maze.train.trainers.imitation.bc_runners import BCDevRunner, BCLocalRunner
from maze.train.trainers.impala.impala_algorithm_config import ImpalaAlgorithmConfig
from maze.train.trainers.impala.impala_runners import ImpalaDevRunner, ImpalaLocalRunner
from maze.train.trainers.ppo.ppo_algorithm_config import PPOAlgorithmConfig
from maze.train.trainers.ppo.ppo_trainer import PPO
from maze.utils.bcolors import BColors


@dataclasses.dataclass
class ConfigurationAuditor:
    """
    Checks specified RunContext configuration for consistency and prepares it for the initialization procedure.
    """

    _run_mode: RunMode
    """Run mode."""
    _args: Dict[str, Any]
    """Arguments passed to RunContext."""

    _kwargs: Optional[Dict[str, Any]] = dataclasses.field(default=None, init=False)
    """Explicitly set keyword arguments."""
    _overrides: Optional[_OverridesType] = dataclasses.field(default=None, init=False)
    """Overrides as dictionary."""
    _ephemeral_init_kwargs: Optional[Dict[str, str]] = dataclasses.field(default=None, init=False)
    """Inferred ephemeral initialization keyword arguments."""

    _EXCLUSIONS = {
        RunMode.TRAINING: {},
        RunMode.ROLLOUT: {"algorithm", "project.name", "critic"}
    }
    """Config attributes to exclude for the corresponding run mode."""

    def audit(self) -> None:
        """
        Audits configuration for inconsistencies and prepares argument for use with Hydra.
        """

        self._kwargs, self._overrides = self._prepare_args_for_hydra_init()
        self._ephemeral_init_kwargs = self._detect_argument_inconsistencies()

    def _prepare_args_for_hydra_init(self) -> Tuple[Dict[str, Any], _OverridesType]:
        """
        Prepares set arguments for Hydra initialization.
        :return: (1) Explicitly set kwargs and (2) overrides prepared for Hydra initialization.
        """

        kwargs = copy.deepcopy(self._args)
        run_dir: str = kwargs.get("run_dir")
        overrides: _OverridesType = self._args["overrides"] if self._args["overrides"] else {}

        # Clear arguments not related to Hydra initialization.
        for arg_to_exclude in ("argv", "overrides", "self", "run_dir", "silent"):
            del kwargs[arg_to_exclude]
        if run_dir is not None:
            overrides["hydra.run.dir"] = run_dir

        # Exclude properties incompatible with specific run modes.
        kwargs = self._filter_run_mode_incompatible_args(kwargs)
        overrides = self._filter_run_mode_incompatible_args(overrides)

        return kwargs, overrides

    def _filter_run_mode_incompatible_args(
        self, items: Union[Dict[str, Any], _OverridesType]
    ) -> Union[Dict[str, Any], _OverridesType]:
        """
        Filters properties incompatible with specific run modes.
        :param items: Dictionary to filter (either explicitly specified arguments or overrides).
        :return: Filtered dictionary.
        """

        return {
            key: val for key, val in items.items()
            if key not in self._EXCLUSIONS[self._run_mode] and all([
                not key.startswith(exclusion + ".") for exclusion in self._EXCLUSIONS[self._run_mode]
            ])
        } if items is not None else {}

    def _detect_argument_inconsistencies(self) -> Dict[str, str]:
        """
        Detects inconsistencies in specified arguments. There are four sources for such inconsistencies:
            - Component types are incompatible with the run mode. This can be the case for runners: DevRunners and
              LocalRunners are exclusively for training, SequentialRunners and ParallelRunners are exclusively for
              Rollout.
            - Codependent components. Some components, e.g. environments and algorithms, are completely independent
              from each other - each environment can be run with each algorithm. Others have a codependency with each
              other, e.g. Runner to Trainer: A runner is specific to a particular trainer. If a component and another
              component to it are specified, these have to be consistent with each other. E.g.: Running
              A2CMultistepTrainer and ESTrainingRunner will fail.
            - Derived config groups. Some config groups depend on other attributes or qualifiers, e.g.
              algorithm_configuration (qualifiers: algorithm, configuration) or algorithm_runner (algorithm, runner).
              Hydra loads these automatically w.r.t. the set qualifiers. This is not the case when RunContext injects
              instantiated objects though, since the value for the underlying Hyra configuration group is not changed
              automatically.
              E.g.: If an A2CAlgorithmConfig is passed, the expected behaviour would be to load the corresponding A2C
              trainer config in algorithm_config. Since the default value for the config group "algorithm" is "ES"
              however, Hydra will load the configuration for the module es-dev or es-local. This results in combining an
              ES trainer with a A2CAlgorithmConfig, which will fail.
            - Nested elements. If a super- and at least one sub-component are specified as instantiated components or
              DictConfigs, RunContext attempts to inject them in the loaded configuration w.r.t. to the respective
              hierarchy levels. This entails that more specialized attributes will overwrite more general ones.
              E.g.: If "model" and "policy" are specified, the model is injected first, then the policy. That way the
              "model" object will overwrite the loaded model and the "policy" object will overwrite the policy in the
              "model" object.
              This doesn't occur in the CLI since no non-primitive value can be passed.
        """

        # I1. Arguments incompatible with run mode.
        self._detect_args_incompatible_with_run_mode()

        # I2. Identify codependent components.
        self._detect_incompatible_codependent_attributes()

        # I3. Extract arguments ensuring derived config groups will load correctly.
        ephemeral_init_kwargs = self._generate_ephemeral_init_kwargs()

        # I4. Check for nested component inconsistency.
        self._detect_nested_component_inconsistency()

        return ephemeral_init_kwargs

    def _detect_args_incompatible_with_run_mode(self) -> None:
        """
        Detect inconsistencies of type 1: Arguments incompatible with run mode.
        """

        illegal_runner_modes = {RunMode.TRAINING: {"parallel", "sequential"}, RunMode.ROLLOUT: {"dev", "local"}}
        if self._kwargs.get("runner"):
            runner_types = self._map_comp_spec_to_conf_module_names("runner", self._kwargs["runner"])["runner"]
            if not all([rt not in illegal_runner_modes[self._run_mode] for rt in runner_types]):
                raise InvalidSpecificationError(
                    "Runner types {rt} not supported in run mode {rm}.".format(
                        rt=illegal_runner_modes[self._run_mode], rm=str(self._run_mode)
                    )
                )

    def _detect_incompatible_codependent_attributes(self) -> None:
        """
        Detect inconsistencies of type 2: Identify codependent components with contradictory settings.
        The only codependent components at this point are algorithm and runner, so we will explicitly check for this
        instead of using some more general and complex approach.
        """

        all_args = {**self._kwargs, **self._overrides}
        if all_args.get("runner") and all_args.get("algorithm"):
            alg_module_names = {
                # Fetch module names for components. If multiple are retrieved, fetch any (algorithm should always be
                # the same).
                comp: self._map_comp_spec_to_conf_module_names(comp, all_args[comp]).get("algorithm")
                for comp in ("algorithm", "runner")
            }

            # If config module names are explicitly specified (via string argument), ephemeral attributes are not
            # necessary.
            if (
                alg_module_names["algorithm"] and alg_module_names["runner"] and
                next(iter(alg_module_names["algorithm"])) != next(iter(alg_module_names["runner"]))
            ):
                raise InvalidSpecificationError(
                    "The specified 'algorithm' object refers to a different algorithm than the specified 'runner' "
                    "object. Please make sure that both components are compatible."
                )

    def _generate_ephemeral_init_kwargs(self) -> Dict[str, str]:
        """
        Prevent inconsistencies of type 3: Instantiated objects not triggering Hydra to load the correct dependent
        config modules.
        To prevent incorrectly resolved config group names, we identify qualifier set as instantiated objects and derive
        the corresponding module name for these. This module name is passed to the Hydra instantiation, after which it
        is removed for the config. In subsequent steps the instantiated qualifer is injected in the loaded Hydra
        configuration, after which it is deleted (hence "ephmeral initialization kwargs").
        :return: Ephemeral initialization keyword arguments.
        """

        ephemeral_init_kwargs: Dict[str, str] = {}
        # "configuration" is a qualifier too, but the API only accepts config module names for it, therefore there is
        # no need to adjust the module loading process.
        for qual_attr in ("algorithm", "model", "launcher", "env", "runner"):
            # Ignore string attributes, since those will specify a module name anyway and thus are not ephemeral.
            if self._kwargs.get(qual_attr) and not isinstance(self._kwargs[qual_attr], str):
                try:
                    module_names = self._map_comp_spec_to_conf_module_names(qual_attr, self._kwargs[qual_attr])
                    assert qual_attr in module_names
                    ambiguous_attrs: Set[str] = set()

                    for arg in module_names:
                        # Ignore if already explicitly set via config module name and hence no ephemeral kwarg
                        # necessary.
                        if type(self._kwargs.get(arg)) in _PrimitiveType.__args__:
                            continue
                        if len(module_names[arg]) == 1:
                            ephemeral_init_kwargs[arg] = next(iter(module_names[arg]))
                        else:
                            ambiguous_attrs.add(arg)

                    if len(ambiguous_attrs):
                        raise InvalidSpecificationError(
                            "Ambiguous mapping for attribute(s) {aa} while resolving qualifying argument {arg}."
                            .format(aa=ambiguous_attrs, arg=qual_attr)
                        )
                except InvalidSpecificationError as error:
                    BColors.print_colored(
                        "Warning: {e} Configuration groups derived from this argument may not be initialized correctly."
                        .format(e=error.args[0]), BColors.WARNING
                    )

        return ephemeral_init_kwargs

    def _detect_nested_component_inconsistency(self) -> None:
        """
        Detect inconsistencies of type 4: Nested components with instantiated parents.
        Setting both super- and subcomponents doesn't work if the corresponding top-level attribute is already
        instantiated - since have no knowledge on internal (initialization) procedures, we can't just replace the
        corresponding sub-component. We thus have to raise an error with such a constellation.
        """

        clean_args = {key: value for key, value in {**self._kwargs, **self._overrides}.items() if value is not None}

        # The attribute proxies reflect hierarchy paths for top level arguments.
        attr_proxies = _ATTRIBUTE_PROXIES[self._run_mode]
        fully_qualified_paths = {
            key: attr_proxies.get(key, {"to": tuple(key.split("."))})["to"]
            for key in ({key for key, val in self._kwargs.items() if val is not None} | self._overrides.keys())
        }
        reverse_fully_qualified_paths = {value: key for key, value in fully_qualified_paths.items()}

        # Only overrides and proxy attributes are interesting for detection of nested ambiguities, since all non-proxy
        # kwargs are top-level components.
        for attr, attr_hpath in fully_qualified_paths.items():
            # Check if parent was set explicitly.
            for i in range(len(attr_hpath) - 1):
                parent_path = attr_hpath[:i + 1]
                parent_path_str = ".".join(parent_path)

                # Try to get parent directly.
                parent = clean_args.get(parent_path_str)
                # Try via proxy.
                if parent is None and parent_path in reverse_fully_qualified_paths:
                    parent = clean_args.get(reverse_fully_qualified_paths[parent_path])
                    parent_path_str = reverse_fully_qualified_paths[parent_path]

                if parent and all([not isinstance(parent, tp) for tp in (str, Mapping)]):
                    raise InvalidSpecificationError(
                        "Element '{parent}' was passed as initialized object, this prevents element '{target}' "
                        "to be set explicitly.".format(parent=parent_path_str, target=attr)
                    )

    @classmethod
    def _map_comp_spec_to_conf_module_names(
        cls, arg_name: str, comp_spec: Union[str, omegaconf.DictConfig, Any]
    ) -> Dict[str, Set[str]]:
        """
        Extract possible corresponding config module names from a component specification. Note that an extraction may
        not be possible at all (e.g. for environments: we cannot know which environment class is configured in which
        file - if at all) or in cases only ambiguously.
        Note: This doesn't cover all components, only those relevant for the identification of type 1, 2 and 3
        inconsistencies.
        :param arg_name: Argument name.
        :param comp_spec: Component specification. Either a DictConfig or a instantiated object.
        :return: Dictionary with entailed possible configuration module names.
        """

        if isinstance(comp_spec, str):
            return {arg_name: {comp_spec}}

        if not isinstance(comp_spec, omegaconf.DictConfig) and not isinstance(comp_spec, Dict):
            _class = type(comp_spec)
        else:
            if "_target_" not in comp_spec:
                raise InvalidSpecificationError("Invalid component specification: {cs}".format(cs=comp_spec))
            _class = hydra.utils.get_class(comp_spec["_target_"])

        if arg_name == "algorithm":
            if not issubclass(_class, AlgorithmConfig):
                raise InvalidSpecificationError()
            _class_map: Dict[Type[AlgorithmConfig], Dict[str, Set[str]]] = {
                A2CAlgorithmConfig: {"algorithm": {"a2c"}},
                BCAlgorithmConfig: {"algorithm": {"bc"}},
                ESAlgorithmConfig: {"algorithm": {"es"}},
                ImpalaAlgorithmConfig: {"algorithm": {"impala"}},
                PPOAlgorithmConfig: {"algorithm": {"ppo"}}
            }

            return _class_map[_class]

        elif arg_name == "runner":
            if not issubclass(_class, Runner):
                raise InvalidSpecificationError()
            _class_map: Dict[Type[Runner], Dict[str, Set[str]]] = {
                # ACDevRunner is ambiguous, could also be PPO.
                ACDevRunner: {"algorithm": {"a2c", "ppo"}, "runner": {"dev"}},
                ACLocalRunner: {"algorithm": {"a2c", "ppo"}, "runner": {"local"}},
                BCDevRunner: {"algorithm": {"bc"}, "runner": {"dev"}},
                BCLocalRunner: {"algorithm": {"bc"}, "runner": {"local"}},
                # ESDevRunner is ambiguous, could also be es-local.
                ESDevRunner: {"algorithm": {"es"}, "runner": {"local", "dev"}},
                ImpalaDevRunner: {"algorithm": {"impala"}, "runner": {"dev"}},
                ImpalaLocalRunner: {"algorithm": {"impala"}, "runner": {"local"}}
            }

            # ACRunners are a special case, as they can enact different trainers. They also have training class
            # attribute we can use to resolve ambiguities.
            if issubclass(_class, ACRunner):
                _trainer_class = (
                    comp_spec.trainer_class if isinstance(comp_spec, ACRunner)
                    else hydra.utils.get_class(comp_spec["trainer_class"])
                )
                if _trainer_class == A2C:
                    alg = "a2c"
                elif _trainer_class == PPO:
                    alg = "ppo"
                else:
                    raise InvalidSpecificationError(
                        "ACRunner only supports A2C or PPO as trainers, not {alg}."
                        .format(alg=str(_trainer_class))
                    )

                return {"algorithm": {alg}, "runner": _class_map[_class]["runner"]}
        else:
            raise InvalidSpecificationError(
                "Extracting config module name for argument '{a}' is not supported.".format(a=arg_name)
            )

        return _class_map[_class]

    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        Returns cleaned keyword arguments.
        :return: Cleaned keyword arguments.
        """

        return self._kwargs

    @property
    def overrides(self) -> _OverridesType:
        """
        Returns cleaned overrides.
        :return: Cleaned overrides.
        """

        return self._overrides

    @property
    def ephemeral_init_kwargs(self) -> Dict[str, str]:
        """
        Returns ephemeral initialization keyword arguments.
        :return: Ephemeral initialization keyword arguments.
        """

        return self._ephemeral_init_kwargs
