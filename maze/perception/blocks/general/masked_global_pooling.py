""" Contains a Masked value mean block. """
from typing import Union, List, Sequence, Dict

import numpy as np
import torch

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock


class MaskedGlobalPoolingBlock(PerceptionBlock):
    """A block applying masked global pooling.
    Pooling is applied wrt the mask (in_keys[1]) and the selected pooling function.
    That is, in the forward pass the input tensor 1 is iterated over in the first 2
    dimensions, where the elements are selected based on the mask, before applying the pooling function.

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param pooling_func: Options: {'mean', 'sum', 'max'}.
    :param pooling_dim: The dimension(s) along which the pooling functions get applied.
    """

    @classmethod
    def _masked_sum(cls, input_tensor: torch.Tensor, mask_tensor: torch.Tensor, dim=Union[int, Sequence[int]]) \
            -> torch.Tensor:
        """Compute sum of tensor along certain dimensions considering a given masking tensor.

        :param input_tensor: The input tensor.
        :param mask_tensor: The masking tensor.
        :param dim: The dimension(s) along which to compute the sum.
        :return: The masked sum tensor.
        """
        # expand dimensions of tensor if required
        while mask_tensor.ndimension() < input_tensor.ndimension():
            mask_tensor = mask_tensor.unsqueeze(-1)

        # zero out masked values
        zero_input_tensor = input_tensor * mask_tensor

        # compute masked average
        masked_sum = torch.sum(zero_input_tensor, dim=dim)
        return masked_sum


    @classmethod
    def _masked_mean(cls, input_tensor: torch.Tensor, mask_tensor: torch.Tensor, dim=Union[int, Sequence[int]]) \
            -> torch.Tensor:
        """Compute mean of tensor along certain dimensions considering a given masking tensor.

        :param input_tensor: The input tensor.
        :param mask_tensor: The masking tensor.
        :param dim: The dimension(s) along which to compute the mean.
        :return: The masked mean tensor.
        """

        # expand dimensions of tensor if required
        while mask_tensor.ndimension() < input_tensor.ndimension():
            mask_tensor = mask_tensor.unsqueeze(-1)

        # zero out masked values
        zero_input_tensor = input_tensor * mask_tensor

        # compute masked average
        tot = torch.sum(zero_input_tensor, dim=dim)
        cnt = torch.sum(mask_tensor, dim=dim)
        masked_mean = torch.div(tot, cnt)
        return masked_mean

    @classmethod
    def _masked_max(cls, input_tensor: torch.Tensor, mask_tensor: torch.Tensor, dim=Union[int, Sequence[int]]) \
            -> torch.Tensor:
        """Compute max of tensor along certain dimensions considering a given masking tensor.

        :param input_tensor: The input tensor.
        :param mask_tensor: The masking tensor.
        :param dim: The dimension(s) along which to compute the max.
        :return: The masked max tensor.
        """
        # expand dimensions of tensor if required
        while mask_tensor.ndimension() < input_tensor.ndimension():
            mask_tensor = mask_tensor.unsqueeze(-1)

        # zero out masked values
        inverted_mask_tensor = torch.tensor(1.0).to(mask_tensor.device) - (1.0 * mask_tensor)
        zero_input_tensor = input_tensor + inverted_mask_tensor * np.finfo(np.float32).min

        # compute masked average
        masked_max, _ = torch.max(zero_input_tensor, dim=dim)
        return masked_max

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 pooling_func: str, pooling_dim: Union[int, Sequence[int]]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)

        assert len(in_shapes[0]) >= 2, 'The first shape given to the block should be at least of dim 2, ' \
                                       f'but got {in_shapes[0]}'
        assert len(in_shapes[1]) >= 1, 'The second shape given to the block should be at least of dim 1 ' \
                                       f'but got {in_shapes[1]}'
        assert all([in_shapes[0][idx] == in_shapes[1][idx] for idx in range(len(in_shapes[1]))])
        self._pooling_func_name = pooling_func
        # select appropriate pooling function
        if self._pooling_func_name == "mean":
            self.pooling_func = self._masked_mean
        elif self._pooling_func_name == "sum":
            self.pooling_func = self._masked_sum
        elif self._pooling_func_name == "max":
            self.pooling_func = self._masked_max
        else:
            raise ValueError(f"Pooling function {self._pooling_func_name} is not yet supported!")

        self.pooling_dim = pooling_dim

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the block, iterating over the first 2 dimensions and pooling the rest in dim=0.

        :param block_input: The block's input dictionary.
        :return: The block's output dictionary.
        """

        # check input tensor
        input_tensor = block_input[self.in_keys[0]]
        mask_tensor = block_input[self.in_keys[1]]
        # Enusre that one value is always true for each batch to circumvent nan values
        mask_tensor[..., 0] = 1.0

        # prepare mask tensor
        mask_tensor: torch.Tensor = ~torch.eq(mask_tensor, 0)

        # apply pooling
        rep = self.pooling_func(input_tensor=input_tensor, mask_tensor=mask_tensor, dim=self.pooling_dim)

        return {self.out_keys[0]: rep}

    def __repr__(self):
        txt = f"{self.__class__.__name__}"
        txt += f'\n\tPooling func: {self._pooling_func_name}'
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
