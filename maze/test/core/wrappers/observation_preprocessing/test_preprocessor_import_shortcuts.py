""" Contains tests for the pre-processing wrapper. """
from maze import preprocessors
from maze.core.utils.registry import Registry
from maze.core.wrappers.observation_preprocessing import preprocessors as preprocessors_module
from maze.core.wrappers.observation_preprocessing.preprocessors.base import PreProcessor


def test_preprocessor_import_shortcuts():
    """Tests if all pre-processors have shortcuts added to the preprocessors/__init__.py"""

    # get list of all registered maze preprocessors
    pre_processors_registry = Registry(base_type=PreProcessor,
                                       root_module=preprocessors_module)

    # iterate preprocessors
    for processor in list(pre_processors_registry.__dict__["type_registry"].values()):
        assert hasattr(preprocessors, processor.__name__)
