""" Contains tests for the plain Python examples in the documentation. """
from docs.source.best_practices_and_tutorials.code_snippets import plain_python_training_high_level, \
    plain_python_training_low_level


def test_plain_python_train_lowlevel():
    """ Tests the plain Python training example. """
    assert plain_python_training_low_level.train(n_epochs=1) == 0


def test_plain_python_train_highlevel():
    """ Tests the plain Python training example. """
    assert plain_python_training_high_level.train(n_epochs=1) == 0
