""" Contains LSTM perception blocks. """
from typing import Union, List, Dict, Sequence

import torch
from torch import nn as nn

from maze.core.annotations import override
from maze.core.utils.factory import Factory
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class LSTMBlock(ShapeNormalizationBlock):
    """A block containing multiple subsequent LSTM layers followed by
    a final time-distributed dense layer with explicit non-linearity.

    The block expects the input tensors to have the from (batch-dim, time-dim, feature-dim).

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param hidden_size: The number of features in the hidden state.
    :param num_layers: Number of recurrent layers.
    :param bidirectional: If True, becomes a bidirectional LSTM.
    :param non_lin: The non-linearity to apply after the final layer.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], hidden_size: int, num_layers: int,
                 bidirectional: bool, non_lin: Union[str, type(nn.Module)]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=3, out_num_dims=3)
        self.input_units = self.in_shapes[0][-1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.non_lin = Factory(base_type=nn.Module).type_from_name(non_lin)
        self.output_units = 2 * self.hidden_size if self.bidirectional else self.hidden_size

        # compile network
        self.net = nn.LSTM(input_size=self.input_units, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           bidirectional=self.bidirectional, batch_first=True)
        self.final_dense = nn.Sequential(nn.Linear(in_features=self.output_units, out_features=self.output_units),
                                         self.non_lin())

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface
        """

        # check input tensor
        input_tensor = block_input[self.in_keys[0]]
        assert input_tensor.ndim == self.in_num_dims[0]
        assert input_tensor.shape[-2:] == self.in_shapes[0]

        # forward pass
        output_tensor, _ = self.net(input_tensor)
        output_tensor = self.final_dense(output_tensor)

        # check output tensor
        assert output_tensor.ndim == self.out_num_dims[0]
        assert output_tensor.shape[-1] == self.output_units

        return {self.out_keys[0]: output_tensor}

    def __repr__(self):
        txt = f"{LSTMBlock.__name__}"
        txt += "\n" + f"({self.input_units}->({self.num_layers} x {self.hidden_size}))"
        txt += "\n" + f"->dense({self.hidden_size}, {self.non_lin.__name__}))"
        if self.bidirectional:
            txt += "\nbidirectional"
        txt += f"\nOut Shapes: {self.out_shapes()}"
        return txt
