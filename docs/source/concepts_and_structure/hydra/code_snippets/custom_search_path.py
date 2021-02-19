# Inside your project in: hydra_plugins/add_custom_config_to_search_path.py

"""Hydra plugin to register additional config packages in the search path."""
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class AddCustomConfigToSearchPathPlugin(SearchPathPlugin):
    """Hydra plugin to register additional config packages in the search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Add custom config to search path (part of SearchPathPlugin interface)."""
        search_path.append("project", "pkg://your_project.conf")