"""
Provides functionality for instantiating objects from configuration.
"""
import copy
import importlib
from typing import TypeVar, Type, Dict, Union, Any, List, Mapping, Generic, Tuple, Sequence, Callable

from hydra.utils import instantiate
from omegaconf import open_dict, DictConfig

ConfigType = Union[None, Mapping[str, Any], Any]
"""Shorthand type for configuration corresponding to a single object.

* Instance can be optional (None values supported)
* Instance can be specified by a dictionary, the type specified in the "_target_" attribute
* The object can already be instantiated
"""

CollectionOfConfigType = Union[List[ConfigType], Mapping[Union[str, Type], ConfigType]]
"""Shorthand type for a list or a dictionary of object parameters from the config files. E.g. used to specify
a collection of ActionConversion objects.

Can be either:
* List of config parameter dictionaries
* Dictionary of config parameter dictionaries
* List of instantiated objects
* Dictionary of instantiated objects
"""

BaseType = TypeVar("BaseType")


class Factory(Generic[BaseType]):
    """
    Supports the creation of instances from configuration, that can be plugged into the environments
    (like demand generators or reward schemes).

    :param base_type: A common interface (parent class) of the registered types (e.g. DemandGenerator)
    """

    def __init__(self, base_type: Type[BaseType]):
        self.base_type = base_type

    def instantiate(self, config: Union[BaseType, ConfigType], **kwargs: Any) -> BaseType:
        """Instantiate an object from configuration.

        This is implemented as a thin layer on top of Hydra's instantiate() function, with the following additions

        * Provides type hinting
        * Asserts the returned type matches the `base_type`
        * Throws an error if there the field '_target_' is missing (Hydra's instantiate() return the config object
          in this case)
        * In case `config` is already an instantiated object, immediately return this existing instance (useful in
          the frequent case of a constructor that accepts either a config object or an already instantiated
          object)

        :param config: The configuration as dictionary
        :param kwargs: Additional arguments that are merged with the configuration dictionary (useful to sideload
                       objects that can not conveniently be specified in the config, e.g. a shared RandomState)

        :return The instantiated object, of type `base_type`
        """
        if config is None:
            return None

        if not isinstance(config, DictConfig) and isinstance(config, self.base_type):
            # nothing to do, object is already instantiated
            return config

        assert not isinstance(config, str), f"instantiation from string not supported, please pass as '_target_' arg"
        assert hasattr(config, "__getitem__"), f"unexpected type {type(config)}"
        assert "_target_" in config, f"Can't instantiate, field '_target_' is missing, config={config}"

        if "_" in config:
            config = copy.deepcopy(config)
            # removing the reserved name "_" is only possible by accessing internal Hydra structures
            with open_dict(config):
                del config["_"]

        o = instantiate(config, _recursive_=config.get("_recursive_", False), **kwargs)
        assert isinstance(o, self.base_type), f"{o} is not of type {self.base_type}"

        return o

    def instantiate_collection(self, config: CollectionOfConfigType, **kwargs: Any
                               ) -> Dict[Union[str, int], BaseType]:
        """Instantiates objects specified in a list or dictionary.

        :param config: A list or dictionary of individual configs, passed to `instantiate()`
        :param kwargs: Additional arguments that are merged with the configuration dictionary (useful to sideload
                       objects that can not conveniently be specified in the config, e.g. a shared RandomState)

        :return A dictionary with either integers (in case config is given as list) or strings as keys and the newly
                created instances as values.
        """
        if isinstance(config, Sequence):
            return {idx: self.instantiate(a, **kwargs) for idx, a in enumerate(config)}
        if isinstance(config, Mapping):
            assert "_target_" not in config, "expected a dictionary of config objects, " \
                                             "but received an instance config instead"

            return {k: self.instantiate(v, **kwargs) for k, v in config.items()}

        raise ValueError(f"unexpected collection type {config}")

    @classmethod
    def _split_module_and_class(cls, path) -> Tuple[str, str]:
        """try to split module and class name from path

        :param path: path in the form package1.package2.MyClass
        :return: tuple (path, class_name), e.g. ("package1.package2", "MyClass")
        """
        split_path = path.split(".")
        path = ".".join(split_path[:-1])
        class_name = split_path[-1]

        return path, class_name

    def type_from_name(self, name: Union[str, Type[BaseType]]) -> Type[BaseType]:
        """Import the given module and lookup the callable or class from the module with the correct base type.

        :param name: Fully qualified name including the module path (e.g.
                     ``maze_envs.logistics.property_based_replenishment.env.maze_env.MazeEnv``)
        :return: The one and only callable or class with the given name that derives from base_type.
        """
        if not isinstance(name, str) and issubclass(name, self.base_type):
            return name

        full_name = name

        name, specified_class_name = self._split_module_and_class(name)

        module = importlib.import_module(name)
        obj_type = getattr(module, specified_class_name, None)
        if not obj_type:
            raise ValueError(f"'{specified_class_name}' not found in module '{name}'")

        if self.base_type != Callable and not issubclass(obj_type, self.base_type):
            raise ValueError(f"Class '{full_name}' is not of type {self.base_type}")

        return obj_type
