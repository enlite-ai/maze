""" Contains interfaces and definitions for the perception module. """
from abc import ABC, abstractmethod
from typing import Dict, Union, List, Sequence, Callable

import numpy as np
import torch
from torch import nn as nn


class PerceptionBlock(ABC, nn.Module):
    """Interface for all perception blocks.
    Perception blocks provide a mapping of M input tensors to N output tensors.
    Both input and output tensors are stored in a dictionary with unique keys.

    :param in_keys: Keys identifying the input tensors.
    :param out_keys: Keys identifying the output tensors.
    :param in_shapes: List of input shapes.
    """

    def __init__(self, in_keys: Union[str, List[str]], out_keys: Union[str, List[str]],
                 in_shapes: Union[Sequence[int], List[Sequence[int]]]):
        super().__init__()
        self.in_keys: List[str] = in_keys if isinstance(in_keys, List) else [in_keys]
        self.out_keys: List[str] = out_keys if isinstance(out_keys, List) else [out_keys]
        self.in_shapes: List[Sequence[int]] = in_shapes if isinstance(in_shapes, List) else [in_shapes]

        # Initialize the dummy dict creators. That is a dictionary linking every in_key index to a callable taking no
        #   arguments and returning a random tensor with the correct shape. This dict should be updated in child
        #   classes if special properties have to hold for a given input (e.g. binary values, symmetric structure,..)
        self.dummy_dict_creators: Dict[int, Callable[[], torch.Tensor]] = dict()
        for ii, in_key in enumerate(self.in_keys):
            self.dummy_dict_creators[ii] = self._default_dummy_creator_factory(self.in_shapes[ii])

    @classmethod
    def _default_dummy_creator_factory(cls, in_shape: Sequence[int]) -> Callable[[], torch.Tensor]:
        """Factory for creating a dummy tensor creator function (necessary to circumvent early binding)

        :param in_shape: The in_shape we want to create a tensor for
        :return: A functional taking no arguments returning a tensor
        """

        def default_dummy_creator() -> torch.Tensor:
            """The default dummy dict creator returns a torch tensor with the in_shape of the corresponding input in
                addition to one more dimension for the batch size

                :return: A random tensor with shape corresponding to the input
            """
            return torch.from_numpy(np.random.randn(*([1] + list(in_shape))).astype(np.float32))

        return default_dummy_creator

    @abstractmethod
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of perception block.

        :param block_input: The block's input dictionary.
        :return: The block's output dictionary.
        """

    def out_shapes(self) -> List[Sequence[int]]:
        """Returns the perception block's output shape.

        :return: a list of output shapes.
        """

        # compile dummy input tensor dictionary
        dummy_dict = dict()
        for ii, in_key in enumerate(self.in_keys):
            dummy_dict[in_key] = self.dummy_dict_creators[ii]()

        # perform forward pass
        with torch.no_grad():
            output_dict = self(dummy_dict)

        # extract output shapes
        return [tuple(v.shape[1:]) if v.ndim > 1 else tuple(v.shape[0:])
                for v in output_dict.values()]

    def get_num_of_parameters(self) -> int:
        """Calculates the total number of parameters in the model
        :return: The total number of parameters
        """
        return sum(pp.numel() for pp in self.parameters())
