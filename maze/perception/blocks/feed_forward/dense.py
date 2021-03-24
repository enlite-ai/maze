""" Contains a dense layer block. """
from collections import OrderedDict
from typing import Union, List, Dict, Sequence

import torch
from torch import nn as nn

from maze.core.annotations import override
from maze.core.utils.factory import Factory
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class DenseBlock(ShapeNormalizationBlock):
    """A block containing multiple subsequent dense layers.
    The block expects the input tensors to have the from (batch-dim, feature-dim).

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param hidden_units: List containing the number of hidden units for hidden layers.
    :param non_lin: The non-linearity to apply after each layer.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], hidden_units: List[int],
                 non_lin: Union[str, type(nn.Module)]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=2, out_num_dims=2)
        self.input_units = self.in_shapes[0][-1]
        self.hidden_units = hidden_units
        self.non_lin = Factory(base_type=nn.Module).type_from_name(non_lin)
        self.output_units = self.hidden_units[-1]

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
        assert input_tensor.shape[-1] == self.input_units, f'failed for obs {self.in_keys[0]} because ' \
                                                           f'{input_tensor.shape[-1]} != {self.input_units}'
        # forward pass
        output_tensor = self.net(input_tensor)

        # check output tensor
        assert output_tensor.ndim == self.out_num_dims[0]
        assert output_tensor.shape[-1] == self.output_units

        return {self.out_keys[0]: output_tensor}

    def build_layer_dict(self) -> OrderedDict:
        """Compiles a block-specific dictionary of network layers.
        This could be overwritten by derived layers
        (e.g. to get a 'BatchNormalizedDenseBlock').

        :return: Ordered dictionary of torch modules [str, nn.Module]
        """
        layer_dict = OrderedDict()

        # treat first layer
        layer_dict["linear_0"] = nn.Linear(self.input_units, self.hidden_units[0])
        layer_dict[f"{self.non_lin.__name__}_0"] = self.non_lin()

        # treat remaining layers
        for i, h in enumerate(self.hidden_units[1:], start=1):
            layer_dict[f"linear_{i}"] = nn.Linear(self.hidden_units[i - 1], self.hidden_units[i])
            layer_dict[f"{self.non_lin.__name__}_{i}"] = self.non_lin()

        return layer_dict

    def __repr__(self):
        txt = f"{DenseBlock.__name__}({self.non_lin.__name__})"
        txt += "\n\t" + f"({self.input_units}->" + "->".join([f"{h}" for h in self.hidden_units]) + ")"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
