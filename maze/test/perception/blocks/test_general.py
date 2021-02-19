""" Unit tests for general perception blocks. """
from typing import Dict

import numpy as np
import torch
from torch import nn as nn

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.action_masking import ActionMaskingBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.general.correlation import CorrelationBlock
from maze.perception.blocks.general.flatten import FlattenBlock
from maze.perception.blocks.general.gap import GlobalAveragePoolingBlock
from maze.perception.blocks.general.slice import SliceBlock
from maze.test.perception.perception_test_utils import build_multi_input_dict, build_input_dict


def test_concat_block():
    """ perception test """

    for d0, d1 in [([100, 16], [100, 16]),
                   ([100, 1, 16], [100, 1, 16])]:
        in_dict = build_multi_input_dict(dims=[d0, d1])
        net = ConcatenationBlock(in_keys=["in_key_0", "in_key_1"], out_keys="concat",
                                 in_shapes=[d0[1:], d1[1:]], concat_dim=-1)
        str(net)
        out_dict = net(in_dict)

        assert isinstance(out_dict, Dict)
        assert out_dict["concat"].shape[-1] == 32
        assert net.out_shapes() == [out_dict["concat"].shape[1:]]


def test_mlp_and_concat():
    """ perception test """

    in_dict = build_multi_input_dict(dims=[[100, 1, 16], [100, 1, 8]])

    feat_dict = dict()
    for in_key, in_tensor in in_dict.items():
        # compile network block
        net = DenseBlock(in_keys=in_key, out_keys=f"{in_key}_feat",
                         in_shapes=(in_tensor.shape[-1],), hidden_units=[32, 32], non_lin=nn.ReLU)

        # update output dictionary
        feat_dict.update(net(in_dict))

    net = ConcatenationBlock(in_keys=list(feat_dict.keys()), out_keys="concat",
                             in_shapes=[(32,), (32,)], concat_dim=-1)
    out_dict = net(feat_dict)
    assert out_dict["concat"].ndim == 3
    assert out_dict["concat"].shape[-1] == 64


def test_global_avg_pooling_block():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 3, 64, 64])
    net: GlobalAveragePoolingBlock = GlobalAveragePoolingBlock(in_keys="in_key", out_keys="out_key",
                                                               in_shapes=[(3, 64, 64)])
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (100, 3)


def test_flatten_block():
    """ perception test """
    in_dict = build_input_dict(dims=[100, 3, 64, 64])
    net: FlattenBlock = FlattenBlock(in_keys="in_key", out_keys="out_key", in_shapes=[(3, 64, 64)], num_flatten_dims=3)
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert set(net.out_keys).issubset(set(out_dict.keys()))
    assert out_dict[net.out_keys[0]].shape == (100, 3 * 64 * 64)


def test_slice_block():
    """ perception test """
    for dims in [[32, 16], [100, 32, 16], [100, 5, 32, 16]]:
        in_dict = build_input_dict(dims=dims)

        net = SliceBlock(in_keys="in_key", out_keys="out_key", in_shapes=(32, 16), slice_dim=-2, slice_idx=-1)
        str(net)
        out_dict = net(in_dict)

        assert isinstance(out_dict, Dict)
        assert net.out_shapes() == [out_dict["out_key"].shape[-1:]]


def test_correlation_block():
    """ perception test """

    for d0, d1 in [([100, 16], [100, 16]),
                   ([100, 5, 16], [100, 5, 16]),
                   ([100, 5, 16], [100, 16]),
                   ([100, 16], [100, 5, 16]),
                   ([100, 25, 16], [100, 10, 16])]:
        in_dict = build_multi_input_dict(dims=[d0, d1])

        # with out reduction
        net = CorrelationBlock(in_keys=["in_key_0", "in_key_1"], out_keys="correlation",
                               in_shapes=[d0[1:], d1[1:]], reduce=False)
        str(net)
        out_dict_no_reduce = net(in_dict)

        assert isinstance(out_dict_no_reduce, Dict)
        assert out_dict_no_reduce["correlation"].shape[-1] == 16
        assert net.out_shapes() == [out_dict_no_reduce["correlation"].shape[1:]]

        # with reduction
        net = CorrelationBlock(in_keys=["in_key_0", "in_key_1"], out_keys="correlation",
                               in_shapes=[d0[1:], d1[1:]], reduce=True)
        str(net)
        out_dict_reduce = net(in_dict)

        assert isinstance(out_dict_reduce, Dict)
        assert (out_dict_reduce["correlation"].ndim + 1) == out_dict_no_reduce["correlation"].ndim


def test_action_masking_block():
    """ perception test """

    # prepare block input
    logits = torch.from_numpy(np.random.randn(32, 5).astype(np.float32))
    mask = np.zeros(shape=(32, 5), dtype=np.float32)
    for i in range(mask.shape[0]):
        for j in np.random.choice(mask.shape[1], size=3):
            mask[i, j] = 1.0
    mask = torch.from_numpy(mask)

    in_dict = {"logits": logits, "mask": mask}

    # with out reduction
    net = ActionMaskingBlock(in_keys=["logits", "mask"], out_keys="masked",
                             in_shapes=[logits.shape[1:], mask.shape[1:]], num_actors=1,
                             num_of_actor_actions=None)
    str(net)
    out_dict_no_reduce = net(in_dict)

    assert isinstance(out_dict_no_reduce, Dict)
    assert out_dict_no_reduce["masked"].shape[-1] == 5
    assert net.out_shapes() == [out_dict_no_reduce["masked"].shape[1:]]


def test_action_masking_block_multi():
    """ perception test """

    # prepare block input
    logits = torch.from_numpy(np.random.randn(32, 4).astype(np.float32))
    mask = np.zeros(shape=(32, 2, 1), dtype=np.float32)
    for i in range(mask.shape[0]):
        for j in np.random.choice(mask.shape[1], size=3):
            mask[i, j] = 1.0
    mask = torch.from_numpy(mask)

    in_dict = {"logits": logits, "mask": mask}

    # with out reduction
    net = ActionMaskingBlock(in_keys=["logits", "mask"], out_keys=["masked_0", 'masked_1'],
                             in_shapes=[logits.shape[1:], mask.shape[1:]], num_actors=2,
                             num_of_actor_actions=2)
    str(net)
    out_dict = net(in_dict)

    assert isinstance(out_dict, Dict)
    assert len(out_dict.keys()) == 2
    assert out_dict["masked_0"].shape[-1] == 2
    assert out_dict["masked_1"].shape[-1] == 2
    assert net.out_shapes() == [out_dict["masked_0"].shape[1:], out_dict["masked_1"].shape[1:]]
