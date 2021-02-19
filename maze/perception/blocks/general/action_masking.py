""" Contains action masking blocks. """
from typing import Union, List, Dict, Sequence

import numpy as np
import torch

from maze.core.annotations import override
from maze.perception.blocks.base import PerceptionBlock


class ActionMaskingBlock(PerceptionBlock):
    """An action masking block.

    The block takes two keys as input where the first key contains the logits tensor and the second key contains the
    binary mask tensor. Masking is performed by adding the smallest possible float32 number to the logits where the
    corresponding mask value is False (0.0).

    :param in_keys: Keys identifying the input tensors.
    :param out_keys: Keys identifying the output tensors.
    :param in_shapes: List of input shapes.
    """

    def __init__(self, in_keys: List[str], out_keys: Union[str, List[str]],
                 in_shapes: List[Sequence[int]], num_actors: int, num_of_actor_actions: Union[int, None]):
        super().__init__(in_keys=in_keys, out_keys=out_keys, in_shapes=in_shapes)
        assert len(self.out_keys) == num_actors, f'The number of out_keys should be equal to the number of actors'
        assert len(self.in_keys) == 2
        assert len(self.in_shapes) == 2

        self.num_actors = num_actors
        self.num_of_actor_actions = num_of_actor_actions

    @override(PerceptionBlock)
    def forward(self, block_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.perception.blocks.base.PerceptionBlock` interface
        """
        action_logits = block_input[self.in_keys[0]]
        action_mask = block_input[self.in_keys[1]]

        if self.num_actors == 1:
            return self._forward_single(action_logits, action_mask)
        else:
            return self._forward_multi(action_logits, action_mask)

    def _forward_multi(self, action_logits: torch.Tensor, action_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the forward pass for a single actor.

        :param action_logits: the action logits
        :param action_mask: the actions mask
        """
        actor_actions_dict = dict()
        for actor_idx in range(self.num_actors):
            start_idx = actor_idx * self.num_of_actor_actions
            end_idx = start_idx + self.num_of_actor_actions

            mask = action_mask[..., actor_idx, :]
            inverted_mask = torch.tensor(1.0).to(mask.device) - mask
            machine_logits = action_logits[..., start_idx:end_idx] + inverted_mask * np.finfo(np.float32).min

            actor_actions_dict[self.out_keys[actor_idx]] = machine_logits

        return actor_actions_dict

    def _forward_single(self, action_logits: torch.Tensor, action_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the forward pass for multiple actors.

        :param action_logits: the action logits
        :param action_mask: the actions mask
        """
        assert action_logits.shape == action_mask.shape, f'action_logtis.shape: {action_logits.shape} vs ' \
                                                         f'action_mask.shape: {action_mask.shape}'

        # forward pass
        inverted_mask = torch.tensor(1.0).to(action_mask.device) - action_mask
        logits_tensor_out = action_logits + inverted_mask * np.finfo(np.float32).min

        return {self.out_keys[0]: logits_tensor_out}

    def __repr__(self):
        txt = f"{self.__class__.__name__}"
        txt += f"\n\tnum_actors: {self.num_actors}"
        if self.num_actors > 1:
            txt += f"\n\tnum_actor_actions: {self.num_of_actor_actions}"
        txt += f"\n\tOut Shapes: {self.out_shapes()}"
        return txt
