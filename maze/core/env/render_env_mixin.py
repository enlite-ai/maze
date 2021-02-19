"""Interface for environments to expose render capabilities."""

from abc import ABC, abstractmethod


class RenderEnvMixin(ABC):
    """Interface for rendering functionality in environments (compatible with gym env).

    Intended to be combined with :obj:'~maze.core.env.base_env.BaseEnv` and potentially other environment interfaces by
    multiple inheritance. e.g. `class MyEnv(BaseEnv, RenderEnvMixin)`.
    """

    @abstractmethod
    def render(self, mode: str = 'human') -> None:
        """Render current state of the environment.

        :param: mode: the render mode.
        """
