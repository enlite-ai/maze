""" Contains tests for the plain Python examples in the documentation. """
from docs.source.best_practices_and_tutorials.code_snippets.plain_python_training import train


def test_plain_python_train():
    """ Tests the plain Python training example. """
    assert train(n_epochs=1) == 0
