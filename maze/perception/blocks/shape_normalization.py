""" Contains shape normalization blocks. """
from abc import ABC, abstractmethod
from typing import Dict, Union, List, Tuple, Sequence

import numpy as np
import torch

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock


class ShapeNormalizationBlock(PerceptionBlock, ABC):
    """Perception block normalizing the input and de-normalizing the output tensor dimensions.

    Examples where this interface needs to be implemented are Dense Layers (batch-dim, feature-dim)
    or Convolution Blocks (batch-dim, feature-dim, row-dim, column-dim)

    :param in_keys: Keys identifying the input tensors.
    :param out_keys: Keys identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param in_num_dims: Required number of dimensions for corresponding input.
    :param out_num_dims: Required number of dimensions for corresponding output.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 in_num_dims: Union[int, List[int]], out_num_dims: Union[int, List[int]]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)
        self.in_num_dims: List[int] = in_num_dims if isinstance(in_num_dims, List) else [in_num_dims]
        self.out_num_dims: List[int] = out_num_dims if isinstance(out_num_dims, List) else [out_num_dims]

    @abstractmethod
    def normalized_forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Shape normalized forward pass called in the actual forward pass of this block.

        :param block_input: The block's shape normalized input dictionary.
        :return: The block's shape normalized output dictionary.
        """

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface
        """

        # normalize input tensors
        normalized_block_input, original_in_batch_shape = self._normalize(block_input)

        # forward pass on shape normalized input
        out_dict: Dict[str, torch.Tensor] = self.normalized_forward(normalized_block_input)

        # reshape output tensors to original batch dimensions
        return self._de_normalize(out_dict, original_in_batch_shape)

    def _normalize(self, tensor_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Sequence[int]]:
        """Normalizes the tensor dictionary to the requested dimensions.
        """

        # prepare all input tensors
        normalized_block_input = dict()
        original_shapes = dict()
        original_in_batch_shapes = []

        for i, in_key in enumerate(self.in_keys):

            # collect original input shape
            in_tensor = tensor_dict[in_key]
            original_shapes[in_key] = list(in_tensor.shape)
            original_num_dims = in_tensor.ndim

            # reshape to target shape
            required_num_dims = self.in_num_dims[i]
            num_feat_dims = required_num_dims - 1
            num_batch_dims = original_num_dims - num_feat_dims
            original_batch_shape = original_shapes[in_key][:num_batch_dims]

            # actual reshaping
            if in_tensor.ndim != required_num_dims:
                batch_dim = int(np.prod(original_batch_shape))
                new_shape = [batch_dim] + original_shapes[in_key][-num_feat_dims:]
                in_tensor = in_tensor.reshape(new_shape)

            # keep normalized block input
            normalized_block_input[in_key] = in_tensor
            original_in_batch_shapes.append(original_batch_shape)

        assert all([original_in_batch_shapes[0] == b for b in original_in_batch_shapes[1:]]), \
            'All inputs should have the same batch dimension'
        original_in_batch_shape = original_in_batch_shapes[0]

        return normalized_block_input, original_in_batch_shape

    def _de_normalize(self, tensor_dict: Dict[str, torch.Tensor], original_in_batch_shape: Sequence[int]):
        """De-normalizes the tensor dictionary to the original batch dimensions.
        """

        # extract common batch dimension of input tensors
        original_batch_shape = list(original_in_batch_shape)
        num_batch_dims = len(original_batch_shape)

        # reshape output tensors to original batch dimensions
        for i, out_key in enumerate(self.out_keys):
            out_tensor = tensor_dict[out_key]
            num_feat_dims = self.out_num_dims[i] - 1
            feature_shape = list(out_tensor.shape[-num_feat_dims:])
            new_shape = original_batch_shape + feature_shape
            tensor_dict[out_key] = out_tensor.reshape(new_shape)

            # shape assertions
            assert num_batch_dims + num_feat_dims == tensor_dict[out_key].ndim

        return tensor_dict
