""" Contains a Functional block. """
import inspect
import types
from typing import Union, List, Sequence, Dict, Callable

import torch

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock


class FunctionalBlock(PerceptionBlock):
    """A block applying a custom callable. It processes a tensor or sequence of tensors and returns a tensor or sequence
    of tensors. If the callable has more than one argument the names of the arguments of the function declaration have
    to match the in_keys of the tensors.

    :param in_keys: One key identifying the input tensors.
    :param out_keys: One key identifying the output tensors.
    :param in_shapes: List of input shapes.
    :param func: A simple callable taking a tensor or a sequence of tensors and returning a tensor or a sequence of
        tensors.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]],
                 func: Callable[[Union[torch.Tensor, Sequence[torch.Tensor]]],
                                Union[torch.Tensor, Sequence[torch.Tensor]]]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)
        self.func = func

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the block, applying the callable to the input.

        :param block_input: The block's input dictionary.
        :return: The block's output dictionary.
        """

        # check input tensor
        input_tensors = {key: block_input[key] for key in self.in_keys}
        if len(input_tensors) > 1:
            output_tensors = self.func(**input_tensors)
        else:
            output_tensors = self.func(list(input_tensors.values())[0])

        out = dict()
        if len(self.out_keys) > 1:
            for key, value in zip(self.out_keys, output_tensors):
                out[key] = value
        else:
            out[self.out_keys[0]] = output_tensors

        return out

    def __repr__(self):
        txt = f"{self.__class__.__name__}"
        if isinstance(self.func, types.BuiltinFunctionType):
            func_string = self.func.__name__
        else:
            try:
                func_string = str(inspect.getsource(self.func))
            except OSError:
                # inspect.getsource is not always able to resolve the file
                func_string = self.func.__name__

        # Functional was defined by a declared python function
        if 'def ' in func_string and '\n' in func_string:
            func_string = func_string.split('def ')[-1].split(':')[0]

        txt += f"\n\t := {func_string}"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
