"""Contains a simple script creating an inference graph for our docs."""
from typing import Sequence, Dict

import torch
from torch import nn

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.inference import InferenceGraph, InferenceBlock
from maze.perception.blocks.joint_blocks.vgg_conv_dense import VGGConvolutionDenseBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.test.perception.perception_test_utils import build_input_tensor


def build_multi_input_dict(dims: Sequence[Sequence[int]], names: Sequence[str]) -> Dict[str, torch.Tensor]:
    """build multi input dictionary"""
    input_dict = dict()
    for i, d in enumerate(dims):
        input_dict[names[i]] = build_input_tensor(d)
    return input_dict


def build_perception_dict():
    """ helper function """
    obs_keys = ["obs_inventory", "obs_screen"]
    in_dict = build_multi_input_dict(dims=[[1, 16], [1, 3, 64, 64]],
                                     names=obs_keys)

    perception_dict = dict()

    # --- block ---
    net = DenseBlock(in_keys="obs_inventory", out_keys="obs_inventory_latent",
                     in_shapes=[in_dict["obs_inventory"].shape[-1:]],
                     hidden_units=[32, 32],
                     non_lin=nn.ReLU)
    perception_dict["obs_inventory_latent"] = net

    # --- block ---
    net = VGGConvolutionDenseBlock(in_keys="obs_screen", out_keys="obs_screen_latent",
                                   in_shapes=[in_dict["obs_screen"].shape[-3:]],
                                   hidden_channels=[8, 16, 32],
                                   hidden_units=[32],
                                   non_lin=nn.ReLU,
                                   use_batch_norm_conv=False)
    perception_dict["obs_screen_latent"] = net

    # --- block ---
    net = ConcatenationBlock(in_keys=list(perception_dict.keys()), out_keys="concat",
                             in_shapes=[(32,), (32,)], concat_dim=-1)
    perception_dict["concat"] = net

    # --- block ---
    net = LinearOutputBlock(in_keys=["concat"], out_keys="action_move",
                            in_shapes=[(64,)], output_units=4)
    perception_dict["action_move"] = net

    # --- block ---
    net = LinearOutputBlock(in_keys=["concat"], out_keys="action_use",
                            in_shapes=[(64,)], output_units=16)
    perception_dict["action_use"] = net

    return in_dict, perception_dict


if __name__ == "__main__":
    """ main """

    # compile perception dict
    in_dict, perception_dict = build_perception_dict()

    # compile inference block and predict everything at once
    net = InferenceBlock(in_keys=["obs_inventory", "obs_screen"],
                         out_keys=["action_move", "action_use"],
                         in_shapes=[(16,), (3, 64, 64)],
                         perception_blocks=perception_dict)
    out_dict = net(in_dict)

    # draw inference graph
    graph = InferenceGraph(inference_block=net)
    graph.show(name='Policy Inference Graph', block_execution=False)

    import matplotlib.pyplot as plt
    plt.show(block=True)
