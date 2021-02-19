""" Contains model weight initialization components. """
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from maze.perception.blocks.feed_forward.graph_attention import GraphAttentionLayer
from maze.perception.blocks.feed_forward.graph_conv import GraphConvLayer, GraphConvBlock


def make_normc_initializer(std: float = 1.0) -> Callable[[torch.Tensor], None]:
    """Compiles normc tensor initialization function.

    :param std: The standard deviation.
    :return: The tensor initialization function.
    """

    def initializer(tensor: torch.Tensor) -> None:
        """Initializes the given tensor.

        :param tensor: The torch tensor to be initialized.
        """
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(tensor.data.pow(2).sum(1, keepdim=True))

    return initializer


def make_module_init_normc(std: float = 1.0) -> Callable[[torch.nn.Module], None]:
    """Compiles normc weight initialization function
    initializing module weights with normc_initializer and biases with zeros.

    :param std: The standard deviation.
    :return: The module initialization function.
    """

    def module_init(m: torch.nn.Module) -> None:
        """Initialize module weights with normc_initializer and biases with zeros.

        :param m: the module to initialize.
        """
        if isinstance(m, nn.Linear) or \
                isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) or \
                isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)) or \
                isinstance(m, GraphConvLayer) or isinstance(m, GraphAttentionLayer):

            # initialize weights
            make_normc_initializer(std)(m.weight.data)

            if isinstance(m, GraphAttentionLayer):
                make_normc_initializer(std)(m.weight_a1.data)
                make_normc_initializer(std)(m.weight_a2.data)
            else:
                # initialize biases with zeros
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

        if isinstance(m, GraphConvBlock):
            if m.node_self_importance.requires_grad is True:
                m.node_self_importance.data.fill_(m.node_self_importance_default)

    return module_init


def compute_sigmoid_bias(probability: float) -> float:
    """Compute the bias value for a sigmoid activation function
    such as in multi-binary action spaces (Bernoulli distributions).

    :param probability: The desired selection probability.
    :return: The respective bias value.
    """
    return np.log(probability) - np.log(1.0 - probability)
