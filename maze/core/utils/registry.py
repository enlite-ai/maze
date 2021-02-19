"""
Provides functionality for dynamic registration of classes w.r.t. specified conditions (e.g. from base class, is not
abstract, ...).
"""

import importlib
import inspect
import logging
import os
from abc import ABC
from collections import ChainMap
from types import ModuleType
from typing import TypeVar, Type, Dict, Union, Any, List, Mapping, Generic, Tuple, AnyStr, Iterable, Sequence, Callable

ConfigType = Union[None, str, Mapping[str, Any], Any]
"""Shorthand type for configuration corresponding to a single object.

* Instance can be optional (None values supported)
* Instance type can be specified by a single string
* Instance can be specified by a dictionary, the type specified in the "type" attribute
* The object can already be instantiated
"""

CollectionOfConfigType = Union[List[ConfigType], Mapping[str, ConfigType]]
"""Shorthand type for a list or a dictionary of object parameters from the config files. E.g. used to specify
a collection of ActionConversion objects.

Can be either:
* List of config parameter dictionaries
* Dictionary of config parameter dictionaries
* List of instantiated objects
* Dictionary of instantiated objects
"""

BaseType = TypeVar("BaseType")


class Registry(Generic[BaseType]):
    """
    Supports the creation of different modules that can be plugged into the environments
    (like demand generators or reward schemes) and can instantiate them from parameters read from config files.

    :param root_module: Starting point for search for suitable classes to be registered.
    :param base_type: A common interface (parent class) of the registered types (e.g. DemandGenerator)
    """

    InstanceType = TypeVar("InstanceType")

    def __init__(self,
                 base_type: Type[BaseType],
                 root_module: Union[Any, Iterable[Any]] = ()):
        self.base_type = base_type

        root_modules = root_module
        # create list from single item if necessary
        if not isinstance(root_module, Iterable):
            root_modules = [root_module]

        self.type_registry: Dict[AnyStr, Type[BaseType]] = dict()

        for m in root_modules:
            self.collect_modules(
                root_module=m,
                base_type=base_type)

    def collect_modules(self,
                        root_module: Any,
                        base_type: Type[BaseType]):
        """
        Populates a registry dictionary, by walking the specified root module.

        :param root_module: Starting point for search for suitable classes to be registered.
        :param base_type: Class restriction. Registered classes/modules have to be of type klass.
        :return: A dictionary with class name -> class type for all registered valid sub-classes.
        """

        # Note: Modules are of type ModuleType. PyCharm doesn't recognize this for type hints however, so we rely on Any
        # + type assertion.
        assert isinstance(root_module, ModuleType)

        modules: List[ModuleType] = self._import_modules(root_module, search_recursively=True)

        # Create dictionary of viable wrapper implementations.
        for module in modules:
            for class_name, class_type in inspect.getmembers(module, inspect.isclass):
                if not (
                        # Is class a type of klass?
                        issubclass(class_type, base_type) and
                        # Don't allow abstract classes.
                        ABC not in class_type.__bases__ and
                        # Avoid proxy-importing of suitable classes from other modules where they are not defined, but
                        # imported.
                        class_type.__module__.startswith(root_module.__name__)
                ):
                    continue

                assert class_name[0] == class_name[0].upper(), \
                    f"Class {class_name} of type  {class_type} does not start with an upper case!"

                if class_name in self.type_registry:
                    assert class_type is self.type_registry[class_name], \
                        f"Class {class_name} already contained in  Registry" \
                        f"\nnew:       {class_type}" \
                        f"\ncontained: {self.type_registry[class_name]}"
                self.type_registry[class_name] = class_type

    @classmethod
    def _import_modules(cls, root: ModuleType, search_recursively: bool) -> List[ModuleType]:
        """
        Compiles search paths from specified module-style paths.

        :param root: Root package or module to start search at.
        :param search_recursively: Flag indicating whether to search specified modules recursively.

        :return: List of imported models in viable search paths.
        """

        # Check if root module is only a module and not a package. If so, return immediately.
        if not hasattr(root, "__path__"):
            return [root]

        root_path: str = list(root.__path__)[0]
        return [
            importlib.import_module(
                # Get path to current package/directory.
                (root.__name__ + "." + dirpath[len(root_path) + 1:].replace("/", ".")).strip(".") +
                # Append module/file name.
                "." + os.path.splitext(filename)[0]
            )
            # Gather directories and filenames beginning in specified root package/module.
            for dirpath, dirnames, filenames in (
                # Utilize full recursion tree if search_recursively, otherwise just the first level.
                os.walk(root_path) if search_recursively else [next(os.walk(root_path))]
            )
            # Ignore __pycache__ and other auxiliary items.
            if not dirpath.split("/")[-1].startswith("__")
            # Only consider .py files.
            for filename in filenames if filename.endswith(".py") or filename.endswith(".pyx")
            # Ignore test modules, since we assume they don't hold any relevant definitions and might lead to cyclic
            # dependencies during a dynamic import.
            if not filename.startswith("test_")
        ]

    def __contains__(self, item: str) -> bool:
        """Support the registry ``in`` operation.

        :param item: The module name of the type (maybe a part of or the entire module path).
        :return: True if at least one of the registered types matches the (partial) path.
        """
        return len(self._query(item)) > 0

    def __getitem__(self, item: str) -> Union[Type[BaseType], Callable]:
        """
        Lookup a type by name.

        If a full path to a type or factory function is provided, will try to import it.

        :param item: The module name of the type (maybe a part of or the entire module path).
        :return: The class type.
        """
        matches = self._query(item)
        if len(matches) == 0:
            # try ad-hoc import of a type
            try:
                matches.append(self.class_type_from_module_name(item))
            except ModuleNotFoundError as se:
                if se.name != item:
                    logging.exception(se)
                pass

        if len(matches) == 0:
            # try ad-hoc import of a factory function
            try:
                matches.append(self.callable_from_path(item))
            except ModuleNotFoundError:
                pass

        if len(matches) == 0:
            raise ValueError(f"Module query '{item}' for registry '{self.base_type}' "
                             f"was not found in {self}")

        if len(matches) > 1:
            raise ValueError(f"Module query '{item}' for registry '{self.base_type}' is not unique: {matches}")

        # lookup the matched path and return
        return matches[0]

    def __setitem__(self, key: Union[str, Tuple[str, ...]], value: Type[BaseType]) -> None:
        """Register new type keys

        :param key: The key to register. Can be either a module path as tuple or just a single string name
        :param value: type of the class to register under the passed key
        """
        if isinstance(key, str):
            key = key,

        self.type_registry[key] = value

    def __repr__(self) -> str:
        """Print the registry content as string."""
        return "Registry:\n" + '\n'.join((f"{k} -> {v}" for k, v in self.type_registry.items()))

    @classmethod
    def _split_module_and_class(cls, path) -> Tuple[str, str]:
        """try to split module and class name from path

        :param path: path in the form package1.package2.MyClass
        :return: tuple (path, class_name), e.g. ("package1.package2", "MyClass")
        """
        split_path = path.split(".")
        # recognize class by upper case letter
        path = ".".join(split_path[:-1])
        class_name = split_path[-1]

        return path, class_name

    def _query(self, item: str) -> List[Type[BaseType]]:
        """query the given path and return all matching class types"""
        query_path, class_name = self._split_module_and_class(item)
        query_path = tuple(query_path.split(".")) if len(query_path) else tuple()

        matches = [class_type for path, class_type in self.type_registry.items() if
                   self._is_path_matching(key=path,
                                          query=query_path,
                                          query_class_name=class_name,
                                          class_type=class_type)]

        return matches

    def class_type_from_module_name(self, module_name: Union[str, Type[BaseType]]) -> Type[BaseType]:
        """Import the given module and lookup the class from the module with the correct base type.

        The implementation expects exactly one matching class per module. A ValueError is returned otherwise. If the
        module name is not valid, a ModuleNotFoundError is triggered.

        :param module_name: Absolute module path (e.g. ``maze_envs.logistics.content_based_replenishment.env.maze_env``)
        :return: The one and only class from the given module that derives from base_type.
        """
        matches = []

        if not isinstance(module_name, str) and issubclass(module_name, self.base_type):
            return module_name

        module_name, specified_class_name = self._split_module_and_class(module_name)

        if not len(module_name):
            raise ValueError(f"path '{specified_class_name}' not found in registry")

        module = importlib.import_module(module_name)

        for class_name, class_type in inspect.getmembers(module):
            if specified_class_name and class_name != specified_class_name:
                continue
            matches.append(class_type)

        if len(matches) == 0:
            raise ValueError(f"'{module_name}.{specified_class_name if specified_class_name else '*'} "
                             f"contains no class implementing '{self.base_type}'")

        if len(matches) > 1:
            raise ValueError(
                f"Module '{module_name}' contains more than one class implementing {self.base_type}: {matches}")

        return matches[0]

    @classmethod
    def callable_from_path(cls, path_string: str) -> Callable[..., Any]:
        """Attempt to import a callable from the specified path.

        :param path_string: Path to the callable to import.
        :return: Imported callable.
        """
        path = path_string.split(".")
        if len(path) == 1:
            callable_name = path[0]
            import builtins
            module = builtins
        else:
            callable_name = path[-1]
            module_name = ".".join(path[:-1])
            module = importlib.import_module(module_name)

        if not hasattr(module, callable_name):
            raise ImportError(f"Could not locate '{path_string}'")
        callable_func = getattr(module, callable_name)

        if not callable(callable_func):
            raise ValueError(f"'{path_string}' is not callable")

        return callable_func

    @classmethod
    def _is_path_matching(cls,
                          key: Tuple[AnyStr, ...],
                          query: Tuple[AnyStr, ...],
                          class_type: Type,
                          query_class_name: str) -> bool:
        """Check if there is an intersection at the end of the paths."""
        if len(query) > len(key):
            return False

        # check if class name matches (if specified)
        if query_class_name and query_class_name != class_type.__name__:
            return False

        # special case: empty query, class name check only (above)
        if not len(query):
            return True

        return key[-len(query):] == query

    @classmethod
    def build_obj(cls, klass_or_callable: Union[Type[BaseType], Callable],
                  instance_or_config: ConfigType = None, **kwargs) -> BaseType:
        """
        Given a class, init an instance of that class with given keyword arguments.

        :param klass_or_callable: Class to instantiate, or alternatively a function returning instance of the
                                  registry base type class.
        :param instance_or_config: Either already an actual instance of klass or keyword arguments to provide
        :param kwargs: Additional arguments that are merged with the configuration dictionary (useful to sideload
                       objects that can not conveniently be specified in the config, e.g. a shared RandomState)

        :return: Instance of the given class
        """

        # If arg was passed in as already instantiated object, just return it
        if isinstance(klass_or_callable, type) and isinstance(instance_or_config, klass_or_callable):
            instance = instance_or_config
            return instance

        # If we have a config dictionary, build the object from it
        config = instance_or_config
        if config is None:
            config = {}

        if "type" in config:
            raise AssertionError(
                f"the configuration contains the type attribute {klass_or_callable}, use Registry.arg_to_obj() to "
                f"construct the object according to this type or remove the 'type' attribute if you prefer "
                f"to specify the type in code.")

        import builtins
        try:
            if klass_or_callable.__module__ == "builtins":
                # builtins don't have named arguments
                instance = klass_or_callable(*ChainMap(config, kwargs).values())
            else:
                instance = klass_or_callable(**ChainMap(config, kwargs))
        except TypeError as e:
            raise ValueError(
                f"failed to instantiate '{klass_or_callable}' with config '{config}' "
                f"and injected arguments {list(kwargs.keys())}") from e

        return instance

    def arg_to_collection(self,
                          arg: CollectionOfConfigType,
                          **kwargs) -> Dict[Union[str, int], BaseType]:
        """Instantiates objects specified in a list or dictionary."""
        if isinstance(arg, Sequence):
            return {idx: self.arg_to_obj(a, **kwargs) for idx, a in enumerate(arg)}
        if isinstance(arg, Mapping):
            assert "type" not in arg, "expected a dictionary of config objects, but received an instance config instead"

            return {k: self.arg_to_obj(v, **kwargs) for k, v in arg.items()}

        raise ValueError(f"unexpected collection type {arg}")

    def arg_to_obj(self,
                   arg: ConfigType,
                   config: Mapping[str, Any] = None,
                   **kwargs) -> BaseType:
        """
        Converts arg (usually passed to constructor of an env) to an instantiated class.
         - If arg is already instantiated, just returns it
         - If arg is a string, then construct an instance according to given type_registry and config parameters
         - If arg is a dict-like configuration, construct a new instance. The type is identified by the reserved
           attribute ``type`` and the remaining attributes are passed to the constructor.

        :param arg: Either
                    - an instantiated object inheriting from base_type
                    - a string, e.g. 'static', usually together with the ``config`` argument
                    - a dict-like configuration, specifying the type name in the
                    reserved attribute ``type`` together with the constructor arguments.
                    (e.g. ``{ 'class': 'static_demand', 'constructor_argument': 1, ... }``)
        :param config: Config to pass to the constructor if arg is a string.
        :param kwargs: Additional arguments that are merged with the configuration dictionary (useful to sideload
                       objects that can not conveniently be specified in the config, e.g. a shared RandomState)

        :return: arg if already instantiated, new object otherwise (see the `build_obj` method above)
        """

        # If arg was passed in as already instantiated object, just return it
        if isinstance(arg, self.base_type):
            return arg

        # Support the creation of None objects.
        if arg is None:
            return None

        # If we have a string, build an object from it using the registry
        if isinstance(arg, str):
            target_type = arg
        # If we got a dict-like structure as argument, extract the target type name from the map and build
        # the object from the registry
        elif isinstance(arg, Mapping):
            assert "type" in arg, f"object cannot be instantiated, 'type' specification missing {arg}"
            target_type = arg["type"]
            config = dict(arg)
            del config["type"]

            # remove a "reserved" key to allow for auxiliary definitions in the config file that do not break the
            # constructor call
            if "_" in config:
                del config["_"]

        # Any other types are not allowed
        else:
            raise TypeError(f"arg should either be an instantiated object of type {str(self.base_type)}, "
                            f"or a dictionary-like structure, or a string, "
                            f"but instead was {str(type(arg))}")

        if isinstance(target_type, str):
            target_class = self[target_type]
        else:
            assert isinstance(target_type, type) and issubclass(target_type, self.base_type), \
                'If the target_type is not a string, it should be a class definition, which is also a subclass of ' \
                'the given base type'
            target_class = target_type

        try:
            # Filter out arguments that are not present in the target_class constructor arguments
            constructor_args = inspect.signature(target_class.__init__).parameters.keys()
            kwargs = {k: v for k, v in kwargs.items() if k in constructor_args}
        except ValueError:
            # inspect.signature does not succeed in certain cases, e.g. builtin functions
            # => Then just skip the filtering step
            pass

        # Create instance of the target_class with config and injected arguments
        instance = self.build_obj(target_class, config, **kwargs)

        # Check if we are returning the expected type -- mainly useful if object was created from a function
        assert isinstance(instance, self.base_type), \
            f"instantiated object {type(instance)} is not an instance of of type {str(self.base_type)}"
        return instance
