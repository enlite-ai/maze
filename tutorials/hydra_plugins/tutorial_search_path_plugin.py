"""Hydra plugin to register additional config packages in the search path."""
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class MazeTutorialSearchPathPlugin(SearchPathPlugin):
    """Hydra plugin to register additional config packages in the search path.

    Be aware that hydra uses an unconventional way to import this object: ``imp.load_module``, which forces a reload
    of this Python module. This breaks the singleton semantics of Python modules and makes it impossible to mock
    this class during testing. Therefore the actual paths are provided by ``loop_envs.__init__``.
    """

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """implement the SearchPathPlugin interface"""
        search_path.append("project", "pkg://tutorial_maze_env.part03_maze_env.conf")
        search_path.append("project", "pkg://tutorial_maze_env.part04_events.conf")
        search_path.append("project", "pkg://tutorial_maze_env.part06_struct_env.conf")
