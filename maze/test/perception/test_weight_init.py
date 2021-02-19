""" Unit tests for weight initialization. """
import torch.nn as nn

from maze.perception.weight_init import compute_sigmoid_bias, make_module_init_normc


def test_compute_sigmoid_bias():
    """ perception test """
    assert compute_sigmoid_bias(0.5) == 0


def test_module_init_normc():
    """ perception test """
    module_init = make_module_init_normc(0.1)
    module = nn.Linear(10, 10)
    module.apply(module_init)
