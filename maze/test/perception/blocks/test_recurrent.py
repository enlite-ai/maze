""" Unit tests for recurrent blocks. """
from typing import Dict

from torch import nn as nn

from maze.perception.blocks.recurrent.lstm import LSTMBlock
from maze.test.perception.perception_test_utils import build_input_dict


def test_lstm_block():
    """ perception test """
    for dims in [[32, 16], [100, 32, 16], [100, 5, 32, 16]]:
        in_dict = build_input_dict(dims=dims)

        net = LSTMBlock(in_keys="in_key", out_keys="out_key", in_shapes=(32, 16), hidden_size=64, num_layers=1,
                        bidirectional=True, non_lin=nn.ReLU)
        str(net)
        out_dict = net(in_dict)

        assert isinstance(out_dict, Dict)
        assert net.out_shapes() == [out_dict["out_key"].shape[-2:]]
        assert list(out_dict["out_key"].shape[:-2]) == dims[:-2]
