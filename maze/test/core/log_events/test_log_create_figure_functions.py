""" Contains log figure unit tests """

import numpy as np
import matplotlib.pyplot as plt
from pytest import raises

from maze.core.log_events.log_create_figure_functions import create_histogram, create_categorical_plot, \
    create_violin_distribution, create_binary_plot


def test_create_histogram():
    """ unit tests """
    fig = create_histogram(np.random.randint(low=0, high=10, size=100))
    plt.close(fig)


def test_create_categorical_plot():
    """ unit test """
    values = [int(i) for i in np.random.randint(low=0, high=10, size=100)]
    fig = create_categorical_plot(values)
    plt.close(fig)

    values = [(v, 10) for v in np.random.randint(low=0, high=10, size=100)]
    fig = create_categorical_plot(values)
    plt.close(fig)

    with raises(NotImplementedError):
        values = np.random.random(100)
        create_categorical_plot(values)


def test_create_multi_binary_plot():

    with raises(NotImplementedError):
        values = [i for i in np.random.randint(low=0, high=2, size=(100, 10))]
        _ = create_binary_plot(values)

    values = [(v, 10) for v in np.random.randint(low=0, high=2, size=(100, 10))]
    fig = create_binary_plot(values)
    plt.close(fig)

    with raises(NotImplementedError):
        values = np.random.random(100)
        create_binary_plot(values)


def test_create_violin_distribution():
    """ unit test """
    values = list(np.random.random(size=(100, 50)))
    fig = create_violin_distribution(values)
    plt.close(fig)
