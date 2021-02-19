"""Shows how to use the custom model composer to build a complex custom policy networks."""
from typing import Dict, Union, Sequence, List

import numpy as np
import torch
import torch.nn as nn

from docs.source.policy_and_value_networks.code_snippets.custom_complex_latent_net import \
    CustomComplexLatentNet
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc


class CustomComplexPolicyNet(nn.Module, CustomComplexLatentNet):
    """Simple feed forward policy network.

    :param obs_shapes: The shapes of all observations as a dict.
    :param action_logits_shapes: The shapes of all actions as a dict structure.
    :param non_lin: The nonlinear activation to be used.
    :param hidden_units: A list of units per hidden layer.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], action_logits_shapes: Dict[str, Sequence[int]],
                 non_lin: Union[str, type(nn.Module)], hidden_units: List[int]):
        nn.Module.__init__(self)
        CustomComplexLatentNet.__init__(self, obs_shapes, non_lin, hidden_units)

        # build action heads
        for action_key, action_shape in action_logits_shapes.items():
            self.perception_dict[action_key] = LinearOutputBlock(
                in_keys='latent', out_keys=action_key, in_shapes=self.perception_dict['latent'].out_shapes(),
                output_units=int(np.prod(action_shape)))

        # build inference block
        in_keys = list(self.obs_shapes.keys())
        self.perception_net = InferenceBlock(
            in_keys=in_keys, out_keys=list(action_logits_shapes.keys()), perception_blocks=self.perception_dict,
            in_shapes=[self.obs_shapes[key] for key in in_keys])

        # apply weight init
        self.perception_net.apply(make_module_init_normc(1.0))
        for action_key in action_logits_shapes.keys():
            self.perception_dict[action_key].apply(make_module_init_normc(0.01))

    def forward(self, in_tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network.

        :param in_tensor_dict: Input tensor dict.
        :return: The computed output of the network.
        """
        return self.perception_net(in_tensor_dict)
