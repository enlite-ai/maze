"""Dummy implementation of a policy net for the dummy env"""

from typing import Dict, Sequence

import torch
import torch.nn as nn

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc
from maze.test.shared_test_utils.dummy_models.base_model import DummyBaseNet


class DummyPolicyNet(DummyBaseNet):
    """Policy network.

    :param obs_shapes: The shapes of all observations as a dict.
    :param action_logits_shapes: The shapes of all actions as a dict structure.
    :param non_lin: The nonlinear activation to be used.
    """

    def __init__(self, obs_shapes: Dict[str, Sequence[int]], action_logits_shapes: Dict[str, Sequence[int]],
                 non_lin: type(nn.Module)):
        super().__init__(obs_shapes, non_lin)

        for action_head_name in action_logits_shapes.keys():
            head_hidden_units = [lambda out_shape: out_shape[0] * 5,
                                 lambda out_shape: out_shape[0] * 2,
                                 lambda out_shape: out_shape[0]]
            head_hidden_units = [func(action_logits_shapes[action_head_name]) for func in head_hidden_units]

            self.perception_dict[f'{action_head_name}_net'] = DenseBlock(
                in_keys='hidden_out', in_shapes=self.perception_dict['hidden_out'].out_shapes(),
                out_keys=f'{action_head_name}_net', hidden_units=head_hidden_units[:-1], non_lin=non_lin)

            self.perception_dict[f'{action_head_name}'] = LinearOutputBlock(
                in_keys=f'{action_head_name}_net',
                in_shapes=self.perception_dict[f'{action_head_name}_net'].out_shapes(),
                out_keys=action_head_name, output_units=head_hidden_units[-1]
            )

        # Set up inference block
        self.perception_net = InferenceBlock(
            in_keys=list(self.obs_shapes.keys()), out_keys=list(action_logits_shapes.keys()),
            in_shapes=[self.obs_shapes[key] for key in self.obs_shapes.keys()],
            perception_blocks=self.perception_dict)

        self.perception_net.apply(make_module_init_normc(1.0))
        for action_head_name in action_logits_shapes.keys():
            self.perception_dict[f'{action_head_name}'].apply(make_module_init_normc(0.01))

    def forward(self, xx: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute forward pass through the network.

        :param xx: Input dict.
        :return: The computed output of the network.
        """
        return self.perception_net(xx)
