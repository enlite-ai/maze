""" Contains a RepeatToMatch block. """
from typing import Union, List, Sequence, Dict

import torch

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock


class RepeatToMatchBlock(PerceptionBlock):
    """A repeat-to-match block.
    This blocks takes two tensors and a dimension index as an input. Then when it's
    forward method is called, it matches the specified the dimension (with :param repeat_at_idx) of the first
    tensor with the specified dimension (:param repeat_at_idx)  of the second tensor. This is done by repeating the
    first tensor n times in dimension in dimension :param repeat_at_idx.
    Here n = tensor_1.shape[repeat_at_idx] - tensor_0.shape[repeat_at_idx]. As a constraint the first tensor has to
    satisfy the following condition: tensor_0[repeat_at_idx] == 1

    :param in_keys: The keys identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param repeat_at_idx: Specify the dimension that should be matched between the tensors.
    """

    def __init__(self, in_keys: List[str], out_keys: Union[str, List[str]],
                 in_shapes: List[Sequence[int]], repeat_at_idx: int):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)

        assert isinstance(in_keys, list), f'in_keys should be a list not {type(in_keys)}'
        assert len(in_keys) == 2, f'number of in keys should be 2 but got {len(in_keys)}'
        if isinstance(out_keys, list):
            assert len(out_keys) == 1, f'number of out keys should be 1 but got {len(out_keys)}'
        else:
            assert isinstance(out_keys, str)
        assert in_shapes[0][repeat_at_idx] == 1

        self.repeat_at_idx = repeat_at_idx

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass, repeat the first tensor to match the second one in the given dimension

        :param block_input: The block's input dictionary.
        :return: The block's output dictionary.
        """

        # check input tensor
        tensor_to_repeat = block_input[self.in_keys[0]]
        assert tensor_to_repeat.shape[self.repeat_at_idx] == 1, f'tensor_0.shape[self.repat_at_idx] should be 1, but ' \
                                                                f'got {tensor_to_repeat.shape[self.repeat_at_idx]} (' \
                                                                f'full shape: {tensor_to_repeat.shape})'

        num_of_repeats = block_input[self.in_keys[1]].shape[self.repeat_at_idx]

        # forward pass
        repeats = list([1 for _ in range(tensor_to_repeat.ndim)])
        repeats[self.repeat_at_idx] = num_of_repeats

        output_tensor = tensor_to_repeat.repeat(*repeats)

        return {self.out_keys[0]: output_tensor}

    def __repr__(self):
        txt = f"{self.__class__.__name__}"
        txt += f"\n\trepeat_at_idx: {self.repeat_at_idx}"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
