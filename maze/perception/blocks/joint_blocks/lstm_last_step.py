""" Contains a joint flattening dense perception block. """
from typing import Union, List, Sequence, Dict

import torch
from torch import nn as nn

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock
from maze.perception.blocks.general.slice import SliceBlock
from maze.perception.blocks.recurrent.lstm import LSTMBlock


class LSTMLastStepBlock(PerceptionBlock):
    """A block containing a LSTM perception block followed by a
    Slicing Block keeping only the output of the final time step.

    For details on flattening see :class:`~maze.perception.blocks.recurrent.lstm.LSTMBlock`.
    For details on dense layers see :class:`~maze.perception.blocks.general.slice.SliceBlock`.

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param hidden_size: The number of features in the hidden state.
    :param num_layers: Number of recurrent layers.
    :param bidirectional: If True, becomes a bidirectional LSTM.
    :param non_lin: The non-linearity to apply after the final layer.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 hidden_size: int, num_layers: int, bidirectional: bool, non_lin: Union[str, type(nn.Module)]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)

        out_keys_lstm = [f"{k}_lstm" for k in self.out_keys]
        self.lstm_block = LSTMBlock(in_keys=self.in_keys, out_keys=out_keys_lstm, in_shapes=self.in_shapes,
                                    hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                                    non_lin=non_lin)

        self.slice_block = SliceBlock(in_keys=out_keys_lstm, out_keys=out_keys,
                                      in_shapes=self.lstm_block.out_shapes(),
                                      slice_dim=-2, slice_idx=-1)

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface
        """

        # forward pass of submodules
        block_output = self.lstm_block(block_input)
        block_output = self.slice_block(block_output)

        return block_output

    def __repr__(self):
        txt = f"{LSTMLastStepBlock.__name__}:"
        txt += f"\n\n{str(self.lstm_block)}"
        txt += f"\n\n{str(self.slice_block)}"
        return txt
