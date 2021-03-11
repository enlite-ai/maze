""" Contains tests for the pre-processing wrapper. """
from maze import preprocessors
from maze.core.wrappers.observation_preprocessing import preprocessors as preprocessors_module
from maze.test.shared_test_utils.helper_functions import all_classes_of_module


def test_preprocessor_import_shortcuts():
    """Tests if all pre-processors have shortcuts added to the preprocessors/__init__.py"""

    # iterate preprocessors
    for processor in all_classes_of_module(preprocessors_module):
        assert hasattr(preprocessors, processor.__name__)
