"""
Loads and post-processes runner configuration for RunContext.
"""
import dataclasses
import os
import sys
from typing import Any, Optional, Mapping, Set, Dict, Sequence

import hydra
import omegaconf
from omegaconf import OmegaConf

from maze.api.utils import RunMode, _PrimitiveType, _ATTRIBUTE_PROXIES, _MISSING_ARGUMENTS, _OverridesType, \
    working_directory, InvalidSpecificationError


@dataclasses.dataclass
class ConfigurationLoader:
    """
    A ConfigurationLoader loads and post-processes a particular configuration for RunContext. This includes injecting
    instantiated objects, resolving proxy arguments etc.
    """

    _run_mode: RunMode
    """Run mode."""
    _kwargs: Dict[str, Any]
    """Explicitly set keyword arguments."""
    _overrides: _OverridesType
    """Overrides as dictionary."""
    _ephemeral_init_kwargs: Dict[str, str]
    """Inferred ephemeral initialization keyword arguments."""

    _workdir: Optional[str] = dataclasses.field(default=None, init=False)
    """Working directory path."""
    _config: Optional[omegaconf.DictConfig] = dataclasses.field(default=None, init=False)
    """Loaded DictConfig object."""
    _init_kwargs: Optional[Dict[str, _PrimitiveType]] = dataclasses.field(default=None, init=False)
    """Keyword arguments used for initialization with Hydra."""
    _inj_kwargs: Optional[Dict[str, Any]] = dataclasses.field(default=None, init=False)
    """Keyword arguments injected into Hydra DictConfig."""
    _set_variables: Set[str] = dataclasses.field(default=None, init=False)
    """Names of set variables."""

    def load(self) -> None:
        """
        Loads Hydra configuration and post-processes it according to specified RunContext.

        This also detects inconsistencies in the specification and raises errors to prevent errors at run time. There
        are the following sources for such inconsistencies:

        * Codependent components. Some components, e.g. environments and algorithms, are completely independent
          from each other - each environment can be run with each algorithm. Others have a codependency with each
          other, e.g. Runner to Trainer: A runner is specific to a particular trainer. If a component and another
          component to it are specified, these have to be consistent with each other. E.g.: Running
          A2CMultistepTrainer and ESTrainingRunner will fail.
        * Derived config groups. Some config groups depend on other attributes or qualifiers, e.g.
          algorithm_configuration (qualifiers: algorithm, configuration) or algorithm_runner (algorithm, runner).
          Hydra loads these automatically w.r.t. the set qualifiers. This is not the case when RunContext injects
          instantiated objects though, since the value for the underlying Hyra configuration group is not changed
          automatically.
          E.g.: If an A2CAlgorithmConfig is passed, the expected behaviour would be to load the corresponding A2C
          trainer config in algorithm_config. Since the default value for the config group "algorithm" is "ES"
          however, Hydra will load the configuration for the module es-dev or es-local. This results in combining an
          ES trainer with a A2CAlgorithmConfig, which will fail.
        * Nested elements. If a super- and at least one sub-component are specified as instantiated components or
          DictConfigs, RunContext attempts to inject them in the loaded configuration w.r.t. to the respective
          hierarchy levels. This entails that more specialized attributes will overwrite more general ones.
          E.g.: If "model" and "policy" are specified, the model is injected first, then the policy. That way the
          "model" object will overwrite the loaded model and the "policy" object will overwrite the policy in the
          "model" object.
          This doesn't occur in the CLI, since no non-primitive value can be passed.

        Furthermore, it resolves proxy arguments w.r.t. the current run mode: Non-top level attributes
        exposed in :py:class:`maze.api.run_context.RunContext` (e.g. "critic").
        """

        # 1. Load Hydra configuration for this algorithm and environment.
        self._load_hydra_config()

        # Change to correct working directory (necessary due to being outside of Hydra scope).
        with working_directory(self._workdir):
            # Allow non-primitives in Hydra config.
            with omegaconf.flag_override(self._config, "allow_objects", True) as cfg:
                OmegaConf.set_struct(cfg, False)

                # 2. Inject instantiated objects.
                self._inject_nonprimitive_instances_into_hydra_config(cfg)

                # 3. Resolve proxy arguments in-place.
                self._resolve_proxy_arguments(cfg)

                # 4. Postprocess loaded configuration.
                self._postprocess_config(cfg)

                # 5. Set up and return runner.
                OmegaConf.set_struct(cfg, True)

    def _postprocess_config(self, cfg: omegaconf.DictConfig) -> None:
        """
        Postprocesses configuration.
        :param cfg: Loaded configuration object.
        """

        # Ensure environment is wrapped by LogStatsWrapper.
        ls_wrapper_cls_name = "maze.core.wrappers.log_stats_wrapper.LogStatsWrapper"
        if ls_wrapper_cls_name not in cfg["wrappers"]:
            cfg["wrappers"][ls_wrapper_cls_name] = {
                "logging_prefix": "train" if self._run_mode == RunMode.TRAINING else "rollout"
            }

    def _load_hydra_config(self) -> None:
        """
        Loads Hydra config.
        :return: DictConfig with arguments fed into Hydra initialization.
        """

        config_loader = self
        proxy_attributes = _ATTRIBUTE_PROXIES[self._run_mode]

        @hydra.main(config_path="../conf", config_name=self._run_mode.value)
        def init_hydra(_cfg: omegaconf.DictConfig) -> None:
            """
            Initializes Hydra configuration with overrides.
            :param _cfg: Initialized DictConfig.
            :return: Initialized DictConfig.
            """

            # It's quite ugly to fetch the config value like this, but it doesn't seem to be possible to make
            # @hydra.main return a value: https://github.com/facebookresearch/hydra/issues/332.
            config_loader._config = _cfg

            # Hydra changes the working directory given a corresponding configuration. This is switched back when
            # leaving the Hydra context though, which we don't want to - hence we grab the working directory path and
            # switch back outside the Hydra context.
            config_loader._workdir = os.getcwd()

        # Gather arguments suitable for @hydra.main.
        self._init_kwargs = {
            # Some arguments, e.g. policy, are not supported by a run mode (for policy: training). Nonetheless we want
            # to be able to process these arguments, as they provide convenience and consistency with the CLI.
            # They are mapped to their equivalents in the run mode configuration itself and afterwards (before
            # initializing the respective runners) removed.
            **{
                (("+" if key in _MISSING_ARGUMENTS[self._run_mode] else "") + key): val
                for key, val in {
                    **(self._kwargs if self._kwargs else {}), **(self._overrides if self._overrides else {})
                }.items()
                if type(val) in _PrimitiveType.__args__
            },
            **self._ephemeral_init_kwargs
        }

        # Resolve proxy attributes dependents, e.g. policy.device to model.policy.device. Limitation: Potential config
        # modules may not be loaded correctly. This seems like a reasonable restriction however, since this is also not
        # supported by Hydra per se.
        init_kwargs_resolutions = {}
        for ik in self._init_kwargs:
            if "." in ik:
                unresolved_path = ik.split(".")
                for i, path in enumerate(unresolved_path[:-1]):
                    if path in proxy_attributes:
                        init_kwargs_resolutions[ik] = ".".join(proxy_attributes[path]["to"]) + "." + \
                                                      ".".join(unresolved_path[i + 1:])
        for ikr_key in init_kwargs_resolutions:
            self._init_kwargs[init_kwargs_resolutions[ikr_key]] = self._init_kwargs.pop(ikr_key)

        # Prepare fake command line arguments and initialize.
        self._set_variables = {key.replace("+", "") for key in self._init_kwargs.keys()}
        sys.argv = [sys.argv[0], *[key + "=" + str(val) for key, val in self._init_kwargs.items()]]
        init_hydra()

        # Remove ephemeral init kwargs.
        for key in self._ephemeral_init_kwargs:
            self._set_variables.remove(key)
            self._init_kwargs.pop(key)

    def _inject_nonprimitive_instances_into_hydra_config(self, cfg: omegaconf.DictConfig) -> Dict[str, Any]:
        """
        Injects non-primitive instances into Hydra configuration.
        :param cfg: DictConfig to inject into.
        :return: Dictionary with injected variables.
        """

        self._inj_kwargs = {
            key: value for key, value in {**self._kwargs, **self._overrides}.items()
            if key not in self._init_kwargs and ("+" + key) not in self._init_kwargs and value is not None
        }
        self._set_variables.update(self._inj_kwargs.keys())

        attr_proxies = _ATTRIBUTE_PROXIES[self._run_mode]

        # Gather instantiated objects to plug into config object after composing it.
        for arg_name in self._inj_kwargs:
            # Overrides can specify nested paths, which have to be accessed recursively.
            if "." in arg_name:
                self._set_value_in_nested_dict(cfg, arg_name.split("."), self._inj_kwargs[arg_name], self._run_mode)
            else:
                if arg_name not in cfg and arg_name not in attr_proxies:
                    raise InvalidSpecificationError("Argument '{arg}' not in configuration.".format(arg=arg_name))
                cfg[arg_name] = self._inj_kwargs[arg_name]

        return self._inj_kwargs

    def _resolve_proxy_arguments(self, cfg: omegaconf.DictConfig) -> None:
        """
        Resolves proxy arguments in configuration.
        Proxy keyword arguments don't reflect top-level properties of the underlying configuration, but act as
        shortcuts to them. Since they are more specific than potentially specified top-level arguments, they replace
        their equivalents in higher-level attributes.
        I.e.: If "model" and "policy" are specified, the "policy" value replaces the existing "model.policy".
        :param cfg: Hydra configuration.
        """

        # Fetch attribute proxies.
        attr_proxies = _ATTRIBUTE_PROXIES[self._run_mode]
        # Resolve set variables to full paths.
        fully_qualified_set_vars = {
            key: ".".join(attr_proxies.get(key, {"to": tuple(key.split("."))})["to"]) for key in self._set_variables
        }
        # Sort to maintain that lower-level, more specific are set after higher-level, more general properties.
        fully_qualified_set_vars_sorted = sorted(fully_qualified_set_vars, key=lambda x: x[1])

        # Resolve proxy attributes.
        for set_var in self._set_variables:
            if set_var in attr_proxies:
                # If variable in question was a string (i.e. a config module name) and this proxy attribute is auto-
                # resolving (see comments for maze.api.utils._ATTRIBUTE_PROXIES), we can ignore this.
                if isinstance(self._kwargs.get(set_var), str) and attr_proxies[set_var]["auto_resolving"]:
                    continue

                # Update value in configuration.
                self._set_value_in_nested_dict(cfg, attr_proxies[set_var]["to"], cfg[set_var], self._run_mode)

                # Re-set sub-properties set at initialization time that may have been overwritten by parent
                # specifications in previous statement.
                for dependent_sv in fully_qualified_set_vars_sorted:
                    if dependent_sv.startswith(fully_qualified_set_vars[set_var]) and dependent_sv in self._init_kwargs:
                        self._set_value_in_nested_dict(
                            cfg, dependent_sv.split("."), self._init_kwargs[dependent_sv], self._run_mode
                        )

                # Remove proxy variable.
                del cfg[set_var]

    @classmethod
    def _set_value_in_nested_dict(
        cls, ndict: Mapping[Any, Any], keys: Sequence[Any], val: Any, run_mode: RunMode
    ) -> None:
        """
        Sets value in nested dict with list of keys.
        :param ndict: Dictionary.
        :param keys: Keys in hierarchical order.
        :param val: Value to set.
        :param run_mode: Current run mode.
        """

        data = ndict
        for k in keys[:-1]:
            data = data[k]

        attr_proxies = _ATTRIBUTE_PROXIES[run_mode]
        if keys[-1] not in data and keys[-1] not in attr_proxies:
            raise InvalidSpecificationError("Argument '{arg}' not in configuration.".format(arg=keys[-1]))
        data[keys[-1]] = val

    @property
    def config(self) -> omegaconf.DictConfig:
        """
        Returns loaded DictConfig.
        :return: Loaded DictConfig.
        """

        return self._config

    @property
    def workdir(self) -> str:
        """
        Returns working directory.
        :return: Working directory.
        """

        return self._workdir
