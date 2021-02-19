"""Shows how to use the custom model composer to build a complex custom value networks."""
from typing import Dict, Union, Sequence, List

import torch
import torch.nn as nn

from docs.source.policy_and_value_networks.code_snippets.custom_complex_latent_net import \
    CustomComplexLatentNet
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc


class CustomComplexCriticNet(nn.Module, CustomComplexLatentNet):
    """Simple feed forward policy network.

    :param obs_shapes: The shapes of all observations as a dict.
    :param non_lin: The nonlinear activation to be used.
    :param hidden_units: A list of units per hidden layer.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]],
                 non_lin: Union[str, type(nn.Module)], hidden_units: List[int]):
        nn.Module.__init__(self)
        CustomComplexLatentNet.__init__(self, obs_shapes, non_lin, hidden_units)

        # build action heads
        self.perception_dict['value'] = LinearOutputBlock(
            in_keys='latent', out_keys='value', in_shapes=self.perception_dict['latent'].out_shapes(),
            output_units=1)

        # build inference block
        in_keys = list(self.obs_shapes.keys())
        self.perception_net = InferenceBlock(
            in_keys=in_keys, out_keys='value', in_shapes=[self.obs_shapes[key] for key in in_keys],
            perception_blocks=self.perception_dict)

        # apply weight init
        self.perception_net.apply(make_module_init_normc(1.0))
        self.perception_dict['value'].apply(make_module_init_normc(0.01))

    def forward(self, in_tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network.

        :param in_tensor_dict: Input tensor dict.
        :return: The computed output of the network.
        """
        return self.perception_net(in_tensor_dict)
