"""Shows how to create a custom cartpole model using no maze perception components."""
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPlainCartpolePolicyNet(nn.Module):
    """Simple feed forward policy network.

    :param obs_shapes: The shapes of all observations as a dict.
    :param action_logits_shapes: The shapes of all actions as a dict structure.
    :param hidden_layer_0: The number of units in layer 0.
    :param hidden_layer_1: The number of units in layer 1.
    :param use_bias: Specify whether to use a bias in the linear layers.
    """
    def __init__(self, obs_shapes: Dict[str, Sequence[int]], action_logits_shapes: Dict[str, Sequence[int]],
                 hidden_layer_0: int, hidden_layer_1: int, use_bias: bool):
        nn.Module.__init__(self)

        self.observation_name = list(obs_shapes.keys())[0]
        self.action_name = list(action_logits_shapes.keys())[0]

        self.l0 = nn.Linear(4, hidden_layer_0, bias=use_bias)
        self.l1 = nn.Linear(hidden_layer_0, hidden_layer_1, bias=use_bias)
        self.l2 = nn.Linear(hidden_layer_1, 2, bias=use_bias)

    def reset_parameters(self) -> None:
        """Reset the parameters of the Model"""

        self.l0.reset_parameters()
        self.l1.reset_parameters()
        self.l1.reset_parameters()

    def forward(self, in_tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network.

        :param in_tensor_dict: Input tensor dict.
        :return: The computed output of the network.
        """
        # Retrieve the observation tensor from the input dict
        xx_tensor = in_tensor_dict[self.observation_name]

        # Compute the forward pass thorough the network
        xx_tensor = F.relu(self.l0(xx_tensor))
        xx_tensor = F.relu(self.l1(xx_tensor))
        xx_tensor = self.l2(xx_tensor)

        # Create the output dictionary with the computed model output
        out = dict({self.action_name: xx_tensor})
        return out
