"""
Registry for wrapper classes.
"""

from typing import Type, Union, Any, Iterable, TypeVar

from maze.core.env.base_env import BaseEnv
from maze.core.utils.registry import Registry, CollectionOfConfigType
from maze.core.wrappers.wrapper import Wrapper

T = TypeVar("T", bound=BaseEnv)


class WrapperRegistry(Registry[Wrapper]):
    """
    Handles dynamic registration of Wrapper sub-classes.
    """

    def __init__(self,
                 root_module: Union[Any, Iterable[Any]] = (),
                 base_type: Type[Wrapper] = Wrapper):
        super().__init__(base_type=base_type,
                         root_module=root_module)

    def wrap_from_config(
            self,
            env: T,
            wrapper_config: CollectionOfConfigType
    ) -> Union[Wrapper, T]:
        """
        Wraps environment in wrappers specified in wrapper_config.

        :param env: Environment to wrap.
        :param wrapper_config: Wrapper specification.
        :return: Wrapped environment of type Wrapper.
        """

        wrapped_env: Union[Wrapper, T] = env
        for wrapper_module in wrapper_config:
            wrapped_env = self[wrapper_module].wrap(
                wrapped_env,
                # Pass on additional arguments
                **wrapper_config[wrapper_module]
            )

        return wrapped_env
