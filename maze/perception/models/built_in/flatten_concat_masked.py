"""Contains a flatten and concatenation masking model applicable in most application scenarios."""
import copy
from typing import Sequence, Dict, List

from torch import nn

from maze.perception.blocks.general.action_masking import ActionMaskingBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.models.built_in.flatten_concat import FlattenConcatBaseNet
from maze.perception.weight_init import make_module_init_normc


class FlattenConcatMaskedPolicyNet(FlattenConcatBaseNet):
    """Masked version of the Flatten and concatenation policy model.

    :param obs_shapes: Dictionary mapping of observation names to shapes.
    :param action_logits_shapes: Dictionary mapping of observation names to shapes.
    :param hidden_units: Dictionary mapping of action names to shapes.
    :param remove_mask_from_obs: Specify to remove the observation from the observation (and only use it for masking).
    :param non_lin: The non-linearity to apply.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 action_logits_shapes: Dict[str, Sequence[int]],
                 hidden_units: List[int],
                 remove_mask_from_obs: bool,
                 non_lin=nn.Module):
        if remove_mask_from_obs:
            new_obs_shapes = copy.deepcopy(obs_shapes)
            for action in action_logits_shapes.keys():
                if action + '_mask' in obs_shapes:
                    del new_obs_shapes[action + '_mask']
        super().__init__(new_obs_shapes, hidden_units, non_lin)

        module_init = make_module_init_normc(std=0.01)
        # build action head
        for action, shape in action_logits_shapes.items():
            if action + '_mask' in obs_shapes:

                self.perception_dict[action + '_logits'] = LinearOutputBlock(
                    in_keys="latent", out_keys=action + '_logits',
                    in_shapes=self.perception_dict["latent"].out_shapes(),
                    output_units=action_logits_shapes[action][-1])
                self.perception_dict[action] = ActionMaskingBlock(
                    in_keys=[f'{action}_logits', f'{action}_mask'], out_keys=action,
                    in_shapes=self.perception_dict[action + '_logits'].out_shapes() + [obs_shapes[f'{action}_mask']],
                    num_actors=1,
                    num_of_actor_actions=None
                )
                self.perception_dict[action + '_logits'].apply(module_init)
            else:
                self.perception_dict[action] = LinearOutputBlock(in_keys="latent", out_keys=action,
                                                                 in_shapes=self.perception_dict["latent"].out_shapes(),
                                                                 output_units=action_logits_shapes[action][-1])
                self.perception_dict[action].apply(module_init)

        # compile inference model
        self.net = InferenceBlock(in_keys=list(obs_shapes.keys()),
                                  out_keys=list(action_logits_shapes.keys()),
                                  in_shapes=list(obs_shapes.values()),
                                  perception_blocks=self.perception_dict)

    def forward(self, x):
        """ forward pass. """
        return self.net(x)
