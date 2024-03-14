""" Import blocks to enable import shortcuts. """
from maze.perception.blocks.base import PerceptionBlock
from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.feed_forward.graph_attention import GraphAttentionBlock
from maze.perception.blocks.feed_forward.graph_conv import GraphConvBlock
from maze.perception.blocks.feed_forward.strided_conv import StridedConvolutionBlock
from maze.perception.blocks.feed_forward.vgg_conv import VGGConvolutionBlock
from maze.perception.blocks.general.action_masking import ActionMaskingBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.general.correlation import CorrelationBlock
from maze.perception.blocks.general.flatten import FlattenBlock
from maze.perception.blocks.general.functional import FunctionalBlock
from maze.perception.blocks.general.gap import GlobalAveragePoolingBlock
from maze.perception.blocks.general.masked_global_pooling import MaskedGlobalPoolingBlock
from maze.perception.blocks.general.multi_index_slicing import MultiIndexSlicingBlock
from maze.perception.blocks.general.repeat_to_match import RepeatToMatchBlock
from maze.perception.blocks.general.self_attention_conv import SelfAttentionConvBlock
from maze.perception.blocks.general.self_attention_seq import SelfAttentionSeqBlock
from maze.perception.blocks.general.slice import SliceBlock
from maze.perception.blocks.joint_blocks.flatten_dense import FlattenDenseBlock
from maze.perception.blocks.joint_blocks.lstm_last_step import LSTMLastStepBlock
from maze.perception.blocks.joint_blocks.strided_conv_dense import StridedConvolutionDenseBlock
from maze.perception.blocks.joint_blocks.vgg_conv_dense import VGGConvolutionDenseBlock
from maze.perception.blocks.joint_blocks.vgg_conv_gap import VGGConvolutionGAPBlock
from maze.perception.blocks.recurrent.lstm import LSTMBlock

# feed forward
assert issubclass(DenseBlock, PerceptionBlock)
assert issubclass(GraphAttentionBlock, PerceptionBlock)
assert issubclass(GraphConvBlock, PerceptionBlock)
assert issubclass(StridedConvolutionBlock, PerceptionBlock)
assert issubclass(VGGConvolutionBlock, PerceptionBlock)

# general
assert issubclass(SliceBlock, PerceptionBlock)
assert issubclass(SelfAttentionSeqBlock, PerceptionBlock)
assert issubclass(SelfAttentionConvBlock, PerceptionBlock)
assert issubclass(RepeatToMatchBlock, PerceptionBlock)
assert issubclass(MultiIndexSlicingBlock, PerceptionBlock)
assert issubclass(MaskedGlobalPoolingBlock, PerceptionBlock)
assert issubclass(GlobalAveragePoolingBlock, PerceptionBlock)
assert issubclass(FunctionalBlock, PerceptionBlock)
assert issubclass(FlattenBlock, PerceptionBlock)
assert issubclass(CorrelationBlock, PerceptionBlock)
assert issubclass(ConcatenationBlock, PerceptionBlock)
assert issubclass(ActionMaskingBlock, PerceptionBlock)

# recurrent
assert issubclass(LSTMBlock, PerceptionBlock)

# joint
assert issubclass(VGGConvolutionDenseBlock, PerceptionBlock)
assert issubclass(LSTMLastStepBlock, PerceptionBlock)
assert issubclass(FlattenDenseBlock, PerceptionBlock)
assert issubclass(StridedConvolutionDenseBlock, PerceptionBlock)
assert issubclass(VGGConvolutionGAPBlock, PerceptionBlock)
