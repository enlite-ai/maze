""" Unit tests for output head perception blocks. """
from typing import Dict

import numpy as np

from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.test.perception.perception_test_utils import build_input_dict


def test_linear_output_block():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 16])
    net: LinearOutputBlock = LinearOutputBlock(in_keys="in_key", out_keys="out_key", in_shapes=(16,), output_units=10)
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert net.output_units == 10
    assert out_dict[net.out_keys[0]].shape[-1] == net.output_units
    assert net.out_shapes() == [out_dict[net.out_keys[0]].shape[-1:]]

    # test bias setting
    net.set_bias(bias=1)
    assert np.allclose(net.net.bias.detach().numpy(), 1)

    net.set_bias(bias=np.full(10, fill_value=2, dtype=np.float32))
    assert np.allclose(net.net.bias.detach().numpy(), 2)
