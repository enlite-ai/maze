""" Contains convolution perception blocks. """
from collections import OrderedDict
from typing import Union, List, Sequence, Dict, Tuple, Optional

import torch
from torch import nn as nn

from maze.core.annotations import override
from maze.core.utils.factory import Factory
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class StridedConvolutionBlock(ShapeNormalizationBlock):
    """A block containing multiple subsequent strided convolution layers.

    One layer consists of a single strided convolution followed by an activation function.
    The block expects the input tensors to have the from (batch-dim, channel-dim, row-dim, column-dim).

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param hidden_channels: List containing the number of hidden channels for hidden layers.
    :param hidden_kernels: List containing the size of the convolving kernels.
    :param non_lin: The non-linearity to apply after each layer.
    :param convolution_dimension: Dimension of the convolution to use [1, 2, 3]
    :param hidden_strides: List containing the strides of the convolutions.
    :param hidden_dilations: List containing the spacing between kernel elements.
    :param hidden_padding: List containing the padding added to both sides of the input
    :param padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'.
    """

    def __init__(self,
                 in_keys: Union[str, List[str]],
                 out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 hidden_channels: List[int],
                 hidden_kernels: List[Union[int, Tuple[int, ...]]],
                 non_lin: Union[str, type(nn.Module)],
                 convolution_dimension: int,
                 hidden_strides: Optional[List[Union[int, Tuple[int, ...]]]],
                 hidden_dilations: Optional[List[Union[int, Tuple[int, ...]]]],
                 hidden_padding: Optional[List[Union[int, Tuple[int, ...]]]],
                 padding_mode: Optional[str]):

        assert convolution_dimension in [1, 2, 3]
        if convolution_dimension == 1:
            self.convolution_nn = nn.Conv1d
            in_out_num_dims = 3
            for hk in hidden_kernels:
                assert isinstance(hk, int)
        elif convolution_dimension == 2:
            self.convolution_nn = nn.Conv2d
            in_out_num_dims = 4
            for hk in hidden_kernels:
                assert isinstance(hk, int) or len(hk) == 2
        else:
            self.convolution_nn = nn.Conv3d
            in_out_num_dims = 5
            for hk in hidden_kernels:
                assert isinstance(hk, int) or len(hk) == 3

        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=in_out_num_dims,
                         out_num_dims=in_out_num_dims)

        self.convolutional_dim = convolution_dimension
        self.input_channels = self.in_shapes[0][-(in_out_num_dims - 1)]

        self.hidden_channels = hidden_channels
        self.hidden_kernels = hidden_kernels
        self.non_lin = Factory(base_type=nn.Module).type_from_name(non_lin)
        self.output_channels = self.hidden_channels[-1]

        # Optional arguments
        num_layers = len(self.hidden_channels)
        self.hidden_strides = hidden_strides if hidden_strides is not None else [1 for _ in range(num_layers)]
        self.hidden_dilations = hidden_dilations if hidden_dilations is not None else [1 for _ in range(num_layers)]
        self.hidden_padding = hidden_padding if hidden_padding is not None else [1 for _ in range(num_layers)]
        self.padding_mode = padding_mode if padding_mode is not None else 'zeros'

        # checks
        assert self.padding_mode in ['zeros', 'reflect', 'replicate', 'circular']
        assert len(self.hidden_channels) == len(self.hidden_kernels)
        assert len(self.hidden_channels) == len(self.hidden_strides)
        assert len(self.hidden_channels) == len(self.hidden_dilations)

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
        assert input_tensor.shape[1] == self.input_channels, f'input_tensor.shape[1] {input_tensor.shape[1]}, ' \
                                                             f'self.input_channels {self.input_channels}'
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
        layer_dict["conv_0"] = self.convolution_nn(in_channels=self.input_channels,
                                                   out_channels=self.hidden_channels[0],
                                                   kernel_size=self.hidden_kernels[0],
                                                   stride=self.hidden_strides[0],
                                                   padding=self.hidden_padding[0],
                                                   dilation=self.hidden_dilations[0],
                                                   padding_mode=self.padding_mode)
        layer_dict[f"{self.non_lin.__name__}_0"] = self.non_lin()

        # treat remaining layers
        for ii in range(1, len(self.hidden_channels)):
            layer_dict[f"conv_{ii}"] = self.convolution_nn(in_channels=self.hidden_channels[ii - 1],
                                                           out_channels=self.hidden_channels[ii],
                                                           kernel_size=self.hidden_kernels[ii],
                                                           stride=self.hidden_strides[ii],
                                                           padding=self.hidden_padding[ii],
                                                           dilation=self.hidden_dilations[ii],
                                                           padding_mode=self.padding_mode)
            layer_dict[f"{self.non_lin.__name__}_{ii}"] = self.non_lin()

        return layer_dict

    def __repr__(self):
        txt = f"{self.__class__.__name__.replace('Conv', f'-{self.convolutional_dim}D-Conv')}({self.non_lin.__name__})"
        txt += f"\n\tFeature Maps: {self.input_channels}->" + "->".join([f"{h}" for h in self.hidden_channels])
        txt += f"\n\tKernel Sizes : " + '->'.join([f'{h}' for h in self.hidden_kernels])
        num_layers = len(self.hidden_kernels)
        if self.hidden_strides != [1 for _ in range(num_layers)]:
            txt += f"\n\tStrides: " + '->'.join([f'{h}' for h in self.hidden_strides])
        if self.hidden_padding != [1 for _ in range(num_layers)]:
            txt += f"\n\tPadding: " + '->'.join([f'{h}' for h in self.hidden_padding])
        if self.hidden_dilations != [1 for _ in range(num_layers)]:
            txt += f"\n\tDilation: " + '->'.join([f'{h}' for h in self.hidden_dilations])
        if self.padding_mode != 'zeros':
            txt += f"\n\tPadding Mode: {self.padding_mode}"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
