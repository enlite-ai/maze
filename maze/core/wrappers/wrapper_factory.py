"""
Registry for wrapper classes.
"""

from typing import Union, TypeVar

from maze.core.env.base_env import BaseEnv
from maze.core.utils.factory import Factory, CollectionOfConfigType
from maze.core.wrappers.wrapper import Wrapper

T = TypeVar("T", bound=BaseEnv)


class WrapperFactory(Factory[Wrapper]):
    """
    Handles dynamic registration of Wrapper sub-classes.
    """

    def __init__(self):
        super().__init__(base_type=Wrapper)

    @classmethod
    def wrap_from_config(
            cls,
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
            wrapped_env = Factory(Wrapper).type_from_name(wrapper_module).wrap(
                wrapped_env,
                # Pass on additional arguments
                **wrapper_config[wrapper_module]
            )

        return wrapped_env
