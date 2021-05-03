""" Unit tests for inference blocks. """
import glob

from torch import nn as nn

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.inference import InferenceBlock, InferenceGraph
from maze.test.perception.perception_test_utils import build_multi_input_dict


def build_perception_dict():
    """ helper function """
    in_dict = build_multi_input_dict(dims=[[100, 1, 16], [100, 1, 8]])

    perception_dict = dict()
    for in_key, in_tensor in in_dict.items():
        # compile network block
        net = DenseBlock(in_keys=in_key, out_keys=f"{in_key}_feat",
                         in_shapes=[in_tensor.shape[-1:]], hidden_units=[32, 32], non_lin=nn.ReLU)
        perception_dict[f"{in_key}_feat"] = net

    net = ConcatenationBlock(in_keys=list(perception_dict.keys()), out_keys="concat",
                             in_shapes=[(32,), (32,)], concat_dim=-1)
    perception_dict["concat"] = net

    return in_dict, perception_dict


def test_inference_block():
    """ perception test """
    in_dict, perception_dict = build_perception_dict()

    # compile inference block and predict everything at once
    net = InferenceBlock(in_keys=["in_key_0", "in_key_1"], out_keys="concat",
                         in_shapes=[(1, 16), (1, 8)], perception_blocks=perception_dict)
    out_dict = net(in_dict)
    assert out_dict["concat"].ndim == 3
    assert out_dict["concat"].shape[-1] == 64
    assert net.out_shapes() == [out_dict["concat"].shape[1:]]

    try:
        import pygraphviz

        # draw inference graph
        graph = InferenceGraph(inference_block=net)
        graph.show(name='my_test_net', block_execution=False)
        graph.save(name='my_test_net', save_path='.')
        assert len(glob.glob('*my_test_net*')) == 2
    except ImportError:
        pass
