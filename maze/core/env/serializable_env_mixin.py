"""
Interface for environments to expose internal components for serialization besides the current state object.
Useful e.g. for trajectory data recording.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class SerializableEnvMixin(ABC):
    """
    This interface provides a standard way of exposing environment components whose state should be serialized
    together with the environment state object when for example recording trajectory data.

    Implement this interface if there are additional components in the env besides state that should be serialized.
    """

    @abstractmethod
    def get_serializable_components(self) -> Dict[str, Any]:
        """
        Return all modules that should be serialized as part of the env besides state.

        Important notes:
         - All returned modules should support serialization using pickle. For most objects, this is possible
           out-of-the-box without any special changes. However, there are some notable exceptions like
           event interfaces -- if any of the modules (or their attributes) keeps reference to an abstract object
           like events interface, the `__getstate__` method will need to be overriden to exclude these
           from pickling.

        :return: Dict in the format of { "serializable_module_name": serializable_module }
        """
        raise NotImplementedError
