""" Contains a vgg style convolution block. """
from collections import OrderedDict
from typing import Union, List, Sequence, Dict

import torch
from torch import nn as nn

from maze.core.annotations import override
from maze.core.utils.factory import Factory
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class VGGConvolutionBlock(ShapeNormalizationBlock):
    """A block containing multiple subsequent vgg style convolutions.

    One convolution stack consists of two subsequent 3x3 convolution layers followed by 2x2 max pooling.
    The block expects the input tensors to have the from (batch-dim, channel-dim, row-dim, column-dim).

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param hidden_channels: List containing the number of hidden channels for hidden layers.
    :param non_lin: The non-linearity to apply after each layer.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], hidden_channels: List[int],
                 non_lin: Union[str, type(nn.Module)]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=4, out_num_dims=4)
        self.input_channels = self.in_shapes[0][-3]
        self.hidden_channels = hidden_channels
        self.non_lin = Factory(base_type=nn.Module).type_from_name(non_lin)
        self.output_channels = self.hidden_channels[-1]

        # compile layer dictionary
        layer_dict = self.build_layer_dict()

        # compile network
        self.net = nn.Sequential(layer_dict)

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface
        """

        # check input tensor
        input_tensor = block_input[self.in_keys[0]]
        assert input_tensor.ndim == self.in_num_dims[0]
        assert input_tensor.shape[1] == self.input_channels

        # forward pass
        output_tensor = self.net(input_tensor)

        # check output tensor
        assert output_tensor.ndim == self.out_num_dims[0]
        assert output_tensor.shape[1] == self.output_channels

        return {self.out_keys[0]: output_tensor}

    def build_layer_dict(self) -> OrderedDict:
        """Compiles a block-specific dictionary of network layers.
        This could be overwritten by derived layers
        (e.g. to get a 'BatchNormalizedConvolutionBlock').

        :return: Ordered dictionary of torch modules [str, nn.Module]
        """
        layer_dict = OrderedDict()

        # treat first layer
        layer_dict["conv_00"] = nn.Conv2d(in_channels=self.input_channels,
                                          out_channels=self.hidden_channels[0],
                                          kernel_size=3, stride=1, padding=1)
        layer_dict[f"{self.non_lin.__name__}_00"] = self.non_lin()

        layer_dict["conv_01"] = nn.Conv2d(in_channels=self.hidden_channels[0],
                                          out_channels=self.hidden_channels[0],
                                          kernel_size=3, stride=1, padding=1)
        layer_dict[f"{self.non_lin.__name__}_01"] = self.non_lin()

        layer_dict["pool_01"] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # treat remaining layers
        for i, h in enumerate(self.hidden_channels[1:], start=1):
            layer_dict[f"conv_{i}0"] = nn.Conv2d(in_channels=self.hidden_channels[i - 1],
                                                 out_channels=self.hidden_channels[i],
                                                 kernel_size=3, stride=1, padding=1)
            layer_dict[f"{self.non_lin.__name__}_{i}0"] = self.non_lin()

            layer_dict[f"conv_{i}1"] = nn.Conv2d(in_channels=self.hidden_channels[i],
                                                 out_channels=self.hidden_channels[i],
                                                 kernel_size=3, stride=1, padding=1)
            layer_dict[f"{self.non_lin.__name__}_{i}1"] = self.non_lin()

            layer_dict[f"pool_{i}1"] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        return layer_dict

    def __repr__(self):
        txt = f"{VGGConvolutionBlock.__name__}({self.non_lin.__name__})"
        txt += f"\n(2x 3x3 conv & 2x2 max-pool)"
        txt += f"\nFeature Maps: {self.input_channels}->" + "->".join([f"{h}" for h in self.hidden_channels])
        txt += f"\nOut Shapes: {self.out_shapes()}"
        return txt
