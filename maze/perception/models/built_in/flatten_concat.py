"""Contains a flatten and concatenation model applicable in most application scenarios."""
from typing import Sequence, Dict, List

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.general.flatten import FlattenBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc
from torch import nn


class FlattenConcatBaseNet(nn.Module):
    """Base flatten and concatenation model for policies and critics.

    :param obs_shapes: Dictionary mapping of observation names to shapes.
    :param hidden_units: Dictionary mapping of action names to shapes.
    :param non_lin: The non-linearity to apply.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 hidden_units: List[int],
                 non_lin: nn.Module):
        super().__init__()
        self.hidden_units = hidden_units
        self.non_lin = non_lin

        self.perception_dict = dict()

        # first, flatten all observations
        flat_keys = []
        for obs, shape in obs_shapes.items():
            out_key = f'{obs}_flat'
            flat_keys.append(out_key)
            self.perception_dict[out_key] = FlattenBlock(in_keys=obs, out_keys=out_key, in_shapes=shape,
                                                         num_flatten_dims=len(shape))

        # next, concatenate flat observations
        in_shapes = [self.perception_dict[k].out_shapes()[0] for k in flat_keys]
        self.perception_dict["concat"] = ConcatenationBlock(in_keys=flat_keys, out_keys='concat', in_shapes=in_shapes,
                                                            concat_dim=-1)

        # build perception part
        self.perception_dict["latent"] = DenseBlock(in_keys="concat", out_keys="latent",
                                                    in_shapes=self.perception_dict["concat"].out_shapes(),
                                                    hidden_units=self.hidden_units, non_lin=self.non_lin)

        # initialize model weights
        module_init = make_module_init_normc(std=1.0)
        for key in self.perception_dict.keys():
            self.perception_dict[key].apply(module_init)


class FlattenConcatPolicyNet(FlattenConcatBaseNet):
    """Flatten and concatenation policy model.

    :param obs_shapes: Dictionary mapping of observation names to shapes.
    :param action_logits_shapes: Dictionary mapping of observation names to shapes.
    :param hidden_units: Dictionary mapping of action names to shapes.
    :param non_lin: The non-linearity to apply.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 action_logits_shapes: Dict[str, Sequence[int]],
                 hidden_units: List[int],
                 non_lin=nn.Module):
        super().__init__(obs_shapes, hidden_units, non_lin)

        # build action head
        for action, shape in action_logits_shapes.items():
            self.perception_dict[action] = LinearOutputBlock(in_keys="latent", out_keys=action,
                                                             in_shapes=self.perception_dict["latent"].out_shapes(),
                                                             output_units=action_logits_shapes[action][-1])

            module_init = make_module_init_normc(std=0.01)
            self.perception_dict[action].apply(module_init)

        # compile inference model
        self.net = InferenceBlock(in_keys=list(obs_shapes.keys()),
                                  out_keys=list(action_logits_shapes.keys()),
                                  in_shapes=list(obs_shapes.values()),
                                  perception_blocks=self.perception_dict)

    def forward(self, x):
        """ forward pass. """
        return self.net(x)


class FlattenConcatStateValueNet(FlattenConcatBaseNet):
    """Flatten and concatenation state value model.

    :param obs_shapes: Dictionary mapping of observation names to shapes.
    :param hidden_units: Dictionary mapping of action names to shapes.
    :param non_lin: The non-linearity to apply.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 hidden_units: List[int],
                 non_lin: nn.Module):
        super().__init__(obs_shapes, hidden_units, non_lin)

        # build action head
        self.perception_dict["value"] = LinearOutputBlock(in_keys="latent", out_keys="value",
                                                          in_shapes=self.perception_dict["latent"].out_shapes(),
                                                          output_units=1)

        module_init = make_module_init_normc(std=0.01)
        self.perception_dict["value"].apply(module_init)

        # compile inference model
        self.net = InferenceBlock(in_keys=list(obs_shapes.keys()),
                                  out_keys="value",
                                  in_shapes=list(obs_shapes.values()),
                                  perception_blocks=self.perception_dict)

    def forward(self, x):
        """ forward pass. """
        return self.net(x)
