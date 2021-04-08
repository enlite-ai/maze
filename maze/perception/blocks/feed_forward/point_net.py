""" Contains an implementation of the point net block from https://arxiv.org/abs/1612.00593 and its components. """
from collections import OrderedDict
from typing import Union, List, Dict, Sequence, Optional

import torch
from torch import nn as nn

from maze.core.annotations import override
from maze.core.utils.factory import Factory
from maze.perception.blocks.general.masked_global_pooling import MaskedGlobalPoolingBlock
from maze.perception.blocks.shape_normalization import ShapeNormalizationBlock


class PointNetFeatureTransformNet(nn.Module):
    """Feature Transform Net as proposed in https://arxiv.org/abs/1612.00593. This Module implements three
       convolutional stacks, each consisting of a 1d Convolution (kernel size =1) followed by an optional batch
       normalization and a specified non-linearity. The resulting output of the convolutions is then pooled in the
       point dimension (N) with the specified pooling method. Next two fully connected layers (again with optional
       batch norm and non linearity) are process the now two dimensional data. Finally one fully connected layer is
       applied before reshaping the data into the output format: BxKxK, where B is the batch dimension, N is the number
       of points and K is the number of features. The input to the module should have the shape BxKxN.

       :param num_features: Number of input features (K).
       :param num_points: Number of input points (N).
       :param embedding_dim: The embedding dimension to use (Paper: 1024).
       :param pooling_func_name: A string in ('max', 'mean', 'sum') specifying the pooling function to use. (Paper:
       'max')
       :param use_batch_norm: Specify whether to use batch_norm (like in original paper).
       :param non_lin: The non-linearity to apply after each fully connected layer.
       :param use_masking: Specify whether to use masking.

        """

    def __init__(self, num_features: int, num_points: int, embedding_dim: int, pooling_func_name: str,
                 use_batch_norm: bool, non_lin: Union[str, type(nn.Module)], use_masking: bool):
        super().__init__()

        # Init class variables
        self._use_batch_norm = use_batch_norm
        self._pooling_func_name = pooling_func_name
        self._num_features = num_features
        self._embedding_dim = embedding_dim
        self._num_points = num_points

        # Init convolutions
        self.conv1 = torch.nn.Conv1d(num_features, embedding_dim // 16, 1)
        self.conv2 = torch.nn.Conv1d(embedding_dim // 16, embedding_dim // 8, 1)
        self.conv3 = torch.nn.Conv1d(embedding_dim // 8, embedding_dim, 1)

        # Set up the pooling operation with masking
        self.use_masking = use_masking
        tensor_in_shape = (self._num_points, embedding_dim)
        self.pooling_block = MaskedGlobalPoolingBlock(
            in_keys='in_tensor' if not self.use_masking else ['in_tensor', 'mask_tensor'],
            in_shapes=tensor_in_shape if not self.use_masking else [tensor_in_shape, (self._num_points,)],
            pooling_func=self._pooling_func_name, pooling_dim=-2, out_keys='masking_out'
        )

        # Init fully connected layers
        self.fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.fc2 = nn.Linear(embedding_dim // 2, embedding_dim // 4)
        self.fc3 = nn.Linear(embedding_dim // 4, num_features * num_features)

        # Init batch norm
        if self._use_batch_norm:
            self.bn1 = nn.BatchNorm1d(embedding_dim // 16)
            self.bn2 = nn.BatchNorm1d(embedding_dim // 8)
            self.bn3 = nn.BatchNorm1d(embedding_dim)
            self.bn4 = nn.BatchNorm1d(embedding_dim // 2)
            self.bn5 = nn.BatchNorm1d(embedding_dim // 4)

        # Init non linearity's
        non_lin = Factory(base_type=nn.Module).type_from_name(non_lin)
        self.non_lin_1 = non_lin()
        self.non_lin_2 = non_lin()
        self.non_lin_3 = non_lin()
        self.non_lin_4 = non_lin()
        self.non_lin_5 = non_lin()

    def forward(self, input_tensor, masking_tensor: Optional[torch.Tensor]):
        """Forward pass through the transformer module

        :param input_tensor: Input to the network (BB, KK, NN)
        :param masking_tensor: Optional masking tensor for the pooling operation (BB, NN).
        :return: A transformation matrix of the form (BB, KK, KK)
        """

        batch_size = input_tensor.shape[0]

        # input_tensor: (BB, KK, NN)
        out = self.conv1(input_tensor)
        if self._use_batch_norm and batch_size > 1:
            out = self.bn1(out)
        out = self.non_lin_1(out)

        # out: (BB, embedding_dim // 16, NN)
        out = self.conv2(out)
        if self._use_batch_norm and batch_size > 1:
            out = self.bn2(out)
        out = self.non_lin_2(out)

        # out: (BB, embedding_dim // 8, NN)
        out = self.conv3(out)
        if self._use_batch_norm and batch_size > 1:
            out = self.bn3(out)
        out = self.non_lin_3(out)
        # out: (BB, embedding_dim , NN)

        # Pooling
        # out: (BB, embedding_dim, NN)
        masking_input = {'in_tensor': out.transpose(2, 1)}
        if self.use_masking:
            masking_input['mask_tensor'] = masking_tensor
        out = self.pooling_block(masking_input)['masking_out']
        # output_tensor: (BB, embedding_dim)

        # out: (BB, embedding_dim)
        out = self.fc1(out)
        if self._use_batch_norm and batch_size > 1:
            out = self.bn4(out)
        out = self.non_lin_4(out)

        # out: (BB, embedding_dim//2)
        out = self.fc2(out)
        if self._use_batch_norm and batch_size > 1:
            out = self.bn5(out)
        out = self.non_lin_5(out)

        # out: (BB, embedding_dim//4)
        out = self.fc3(out)
        # out: (BB, num_features ** 2)

        identity = torch.flatten(torch.eye(self._num_features),
                                 start_dim=-2).to(torch.float32).to(out.device)
        identity = identity.repeat(batch_size, 1)
        out = out + identity

        # out: (BB, num_features ** 2)
        out = out.view(-1, self._num_features, self._num_features)
        # out: (BB, num_features, num_features)
        return out

    def get_internal_shape_inference_as_string(self) -> str:
        """Return a string representation of the internal workings of the block.

        :return: A string depicting the input processing done by the block with a focus on the shapes.
        """
        batch_size = 'BB'
        input_shape = f'({batch_size}x{self._num_features}x{self._num_points})'
        conv1_out_shape = f'({batch_size}x{self._embedding_dim // 16}x{self._num_points})'
        conv2_out_shape = f'({batch_size}x{self._embedding_dim // 8}x{self._num_points})'
        conv3_out_shape = f'({batch_size}x{self._embedding_dim}x{self._num_points})'
        pooling_out_shape = f'({batch_size}x{self._embedding_dim})'
        fc1_out_shape = f'({batch_size}x{self._embedding_dim // 2})'
        fc2_out_shape = f'({batch_size}x{self._embedding_dim // 4})'
        fc3_out_shape = f'({batch_size}x{self._num_features * self._num_features})'
        out_shape = f'({batch_size}x{self._num_features}x{self._num_features})'
        return '->'.join([input_shape, conv1_out_shape, conv2_out_shape, conv3_out_shape, pooling_out_shape,
                          fc1_out_shape, fc2_out_shape, fc3_out_shape, out_shape])


class PointNetFeatureBlock(ShapeNormalizationBlock):
    """PointNet block allowing to embed a variable sized set of point observations into a fixed size feature vector via
    the PointNet mechanics. (https://arxiv.org/abs/1612.00593 from Stanford University).

    The block processed the input with one input transformation, two 1d convolutions each followed by an optional batch
    normalization and a non linearity. Then an optional feature transformation is applied before a final convolutional
    layer. Lastly a masked global pooling block is used to pool all values in the point dimension resulting in a
    two dimensional vector (batch_dim, feature_dim). The mask for pooling is an optional parameter.

    :param in_keys: One key identifying the input tensors, a second optional one identifying the masking tensor.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param embedding_dim: The embedding dimension to use throughout the block, this is also specifies the dimension of
        the output. (Paper: 1024)
    :param pooling_func_name: A string in ('max', 'mean', 'sum') specifying the pooling function to use. (Paper: 'max')
    :param use_feature_transform: Whether to use the feature transformation after the second convolution. (Paper: True)
    :param use_batch_norm: Specify whether to use batch_norm (is disables for batches of size <2).
    :param non_lin: The non-linearity to apply after each layer.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]], embedding_dim: int, pooling_func_name: str,
                 use_feature_transform: bool, use_batch_norm: bool, non_lin: Union[str, type(nn.Module)]):

        # Infer number of input dimension depending if mask is provided
        in_keys = in_keys if isinstance(in_keys, List) else [in_keys]
        in_num_dims = 3 if len(in_keys) == 1 else [3, 2]
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes, in_num_dims=in_num_dims,
                         out_num_dims=2)

        # Input parameter assertions: checks if the first input (X) has 2 dimensions (NN, KK).
        assert len(self.in_shapes[0]) == 2
        if len(self.in_keys) == 2:
            # checks that the mask has only one input dimension (NN)
            assert len(self.in_shapes[1]) == 1
            #  checks that the point dimension in X and Mask are the same
            assert self.in_shapes[0][-2] == self.in_shapes[1][-1], f'Point dimension should fit: {self.in_shapes[0]} ' \
                                                                   f'vs {self.in_shapes[1]}'

        # Init class variables
        self._use_feature_transform = use_feature_transform
        self._embedding_dim = embedding_dim
        self._use_batch_norm = use_batch_norm
        self._num_in_features = self.in_shapes[0][-1]
        self._num_points = self.in_shapes[0][-2]
        self._use_masking = len(self.in_keys) > 1

        self.input_transform = PointNetFeatureTransformNet(
            self._num_in_features, self._num_points, non_lin=non_lin, use_batch_norm=self._use_batch_norm,
            embedding_dim=embedding_dim, pooling_func_name=pooling_func_name, use_masking=self._use_masking)
        if self._use_feature_transform:
            self.feature_transform = PointNetFeatureTransformNet(
                embedding_dim // 16, self._num_points, non_lin=non_lin, use_batch_norm=self._use_batch_norm,
                embedding_dim=embedding_dim, pooling_func_name=pooling_func_name, use_masking=self._use_masking
            )

        self.conv1 = torch.nn.Conv1d(self._num_in_features, embedding_dim // 16, 1)
        self.conv2 = torch.nn.Conv1d(embedding_dim // 16, embedding_dim // 8, 1)
        self.conv3 = torch.nn.Conv1d(embedding_dim // 8, embedding_dim, 1)

        if self._use_batch_norm:
            self.bn1 = nn.BatchNorm1d(embedding_dim // 16)
            self.bn2 = nn.BatchNorm1d(embedding_dim // 8)
            self.bn3 = nn.BatchNorm1d(embedding_dim)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        # Set up the pooling operation with masking
        tensor_in_shape = (self._num_points, embedding_dim)
        self.pooling_block = MaskedGlobalPoolingBlock(
            in_keys='in_tensor' if not self._use_masking else ['in_tensor', self.in_keys[1]],
            in_shapes=tensor_in_shape if not self._use_masking else [tensor_in_shape, self.in_shapes[1]],
            pooling_func=pooling_func_name, pooling_dim=-2, out_keys='masking_out'
        )

        self.pooling_func_str = pooling_func_name
        self.non_lin_cls = Factory(base_type=nn.Module).type_from_name(non_lin)
        self.non_lin_1 = self.non_lin_cls()
        self.non_lin_2 = self.non_lin_cls()

        # Debugging parameter for displaying the internal processing of the input in more detail.
        self.print_internal_shape_representation = False

    @override(ShapeNormalizationBlock)
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.shape_normalization.ShapeNormalizationBlock` interface
        """

        # check input tensor
        input_tensor = block_input[self.in_keys[0]]
        mask_tensor = None if not self._use_masking else block_input[self.in_keys[1]]
        assert input_tensor.ndim == self.in_num_dims[0]
        assert input_tensor.shape[-1] == self._num_in_features, f'failed for obs {self.in_keys[0]} because ' \
                                                                f'{input_tensor.shape[-1]} != {self._num_in_features}'
        # forward pass
        # input: (BB, NN, KK)
        input_tensor = input_tensor.transpose(2, 1)
        # input_tensor: (BB, KK, NN)
        input_transformation_matrices = self.input_transform(input_tensor, mask_tensor)
        # input_transformation_matrices: (BB, KK, KK)

        out = torch.bmm(input_transformation_matrices, input_tensor)
        # out: (BB, KK, NN)

        out = self.non_lin_1(self.bn1(self.conv1(out)))
        # out: (BB, embedding_dim // 16, NN)

        if self._use_feature_transform:
            feature_transformation_matrices = self.feature_transform(out, mask_tensor)
            # feature_transformation_matrices: (BB, embedding_dim // 16,  embedding_dim // 16)
            out = torch.bmm(feature_transformation_matrices, out)
            # out: (BB, embedding_dim // 16, NN)

        out = self.non_lin_2(self.bn2(self.conv2(out)))
        # out: (BB, embedding_dim // 8, NN)
        out = self.bn3(self.conv3(out))
        # out: (BB, embedding_dim, NN)

        masking_input = {'in_tensor': out.transpose(2, 1)}
        if self._use_masking:
            masking_input[self.in_keys[1]] = mask_tensor

        output_tensor = self.pooling_block(masking_input)['masking_out']
        # output_tensor: (BB, embedding_dim)

        # check output tensor
        assert output_tensor.ndim == self.out_num_dims[0]
        assert output_tensor.shape[-1] == self._embedding_dim

        return {self.out_keys[0]: output_tensor}

    def _get_internal_shape_inference_as_string(self) -> str:
        """Return a string representation of the internal workings of the block.

        :return: A string depicting the input processing done by the block with a focus on the shapes.
        """
        batch_size = 'BB'
        shapes = OrderedDict()
        in_shape = f'({batch_size}x{self._num_points}x{self._num_in_features})'
        shapes['in_shape'] = in_shape
        shapes['t_shape'] = f'({batch_size}x{self._num_in_features}x{self._num_points})'
        shapes['input_transform_shape'] = self.input_transform.get_internal_shape_inference_as_string()
        shapes['input_transform_out_shape'] = f'({batch_size}x{self._num_in_features}x{self._num_points})'
        shapes['conv1_out_shape'] = f'({batch_size}x{self._embedding_dim // 16}x{self._num_points})'
        if self._use_feature_transform:
            shapes['feature_transform_shape'] = self.feature_transform.get_internal_shape_inference_as_string()
            shapes['feature_transform_out_shape'] = f'({batch_size}x{self._embedding_dim // 16}x{self._num_points})'
        shapes['conv2_out_shape'] = f'({batch_size}x{self._embedding_dim // 8}x{self._num_points})'
        shapes['conv3_out_shape'] = f'({batch_size}x{self._embedding_dim}x{self._num_points})'
        shapes['pooling_out_shape'] = f'({batch_size}x{self._embedding_dim})'
        max_length = max(map(len, shapes.values()))
        shapes_txt = '\n\t' + '-' * (max_length // 2) + '   Internal Shapes inference:   ' + '-' * (max_length // 2)
        max_length = len(shapes_txt)
        for idx, (name, shape) in enumerate(shapes.items()):
            shapes_txt += f'\n\t\t{idx}, {name}: {shape}'
        shapes_txt += '\n\t' + '-' * max_length

        return shapes_txt

    def __repr__(self) -> str:
        """Return a string representation of the block.

        :return: A string representation of the block.
        """

        txt = f"{PointNetFeatureBlock.__name__}({self.non_lin_cls.__name__})"
        txt += f"\n\tembedding_dim: {self._embedding_dim}"
        txt += f"\n\tpooling_func_str: {self.pooling_func_str}"
        txt += f"\n\tuse_feature_transform: {self._use_feature_transform}"
        txt += f"\n\tuse_batch_norm: {self._use_batch_norm}"

        if self.print_internal_shape_representation:
            txt += self._get_internal_shape_inference_as_string()

        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
