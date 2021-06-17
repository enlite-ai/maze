"""Contains a flatten and concatenation model applicable in most application scenarios."""
from typing import Sequence, Dict, List

from torch import nn

from maze.perception.blocks import PerceptionBlock
from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.models.built_in.flatten_concat import FlattenConcatBaseNet
from maze.perception.weight_init import make_module_init_normc


class FlattenConcatSharedEmbeddingPolicyNet(FlattenConcatBaseNet):
    """Flatten and concatenation policy model.

    :param obs_shapes: Dictionary mapping of observation names to shapes.
    :param action_logits_shapes: Dictionary mapping of observation names to shapes.
    :param hidden_units: List of hidden units to use for the embedding.
    :param head_units: List of hidden units to use for the action/value head.
    :param non_lin: The non-linearity to apply.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 action_logits_shapes: Dict[str, Sequence[int]],
                 hidden_units: List[int],
                 head_units: List[int],
                 non_lin=nn.Module):
        super().__init__(obs_shapes, hidden_units, non_lin)

        # build perception part
        self.perception_dict["head"] = DenseBlock(in_keys="latent", out_keys="head",
                                                  in_shapes=self.perception_dict["latent"].out_shapes(),
                                                  hidden_units=head_units, non_lin=self.non_lin)

        self.perception_dict['head'].apply(make_module_init_normc(std=1.0))

        # build action head
        for action, shape in action_logits_shapes.items():
            self.perception_dict[action] = LinearOutputBlock(in_keys="head", out_keys=action,
                                                             in_shapes=self.perception_dict["head"].out_shapes(),
                                                             output_units=action_logits_shapes[action][-1])

            module_init = make_module_init_normc(std=0.01)
            self.perception_dict[action].apply(module_init)

        # compile inference model
        self.net = InferenceBlock(in_keys=list(obs_shapes.keys()),
                                  out_keys=list(action_logits_shapes.keys()) + ['latent'],
                                  in_shapes=list(obs_shapes.values()),
                                  perception_blocks=self.perception_dict)

    def forward(self, x):
        """ forward pass. """
        return self.net(x)


class FlattenConcatSharedEmbeddingStateValueNet(nn.Module):
    """Flatten and concatenation state value model.

    :param obs_shapes: Dictionary mapping of observation names to shapes.
    :param head_units: List of hidden units to use for the action/value head.
    :param non_lin: The non-linearity to apply.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 head_units: List[int],
                 non_lin: nn.Module):
        super().__init__()

        self.perception_dict: Dict[str, PerceptionBlock] = dict()
        # build action head

        # build perception part
        self.perception_dict["head"] = DenseBlock(in_keys="latent", out_keys="head",
                                                  in_shapes=obs_shapes["latent"],
                                                  hidden_units=head_units, non_lin=non_lin)

        self.perception_dict["value"] = LinearOutputBlock(
            in_keys="head", out_keys="value", in_shapes=self.perception_dict["head"].out_shapes(),
            output_units=1)

        self.perception_dict['head'].apply(make_module_init_normc(std=1.0))
        self.perception_dict["value"].apply(make_module_init_normc(std=0.01))

        # compile inference model
        self.net = InferenceBlock(in_keys=list(obs_shapes.keys()),
                                  out_keys="value",
                                  in_shapes=list(obs_shapes.values()),
                                  perception_blocks=self.perception_dict)

    def forward(self, x):
        """ forward pass. """
        return self.net(x)
