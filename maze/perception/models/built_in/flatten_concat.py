"""Contains a flatten and concatenation model applicable in most application scenarios."""
from typing import Sequence, Dict, List, Tuple

import torch
from maze.perception.blocks.general.functional import FunctionalBlock
from maze.train.trainers.common.value_transform import support_to_scalar
from torch import nn

from maze.perception.blocks import PerceptionBlock
from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.general.flatten import FlattenBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc


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

        self.perception_dict: Dict[str, PerceptionBlock] = dict()

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
        self.perception_dict["value"] = LinearOutputBlock(
            in_keys="latent", out_keys="value", in_shapes=self.perception_dict["latent"].out_shapes(),
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


class FlattenConcatCategoricalStateValueNet(FlattenConcatBaseNet):
    """Flatten and concatenation state value model.

    :param obs_shapes: Dictionary mapping of observation names to shapes.
    :param hidden_units: Dictionary mapping of action names to shapes.
    :param non_lin: The non-linearity to apply.
    :param support_range: Tuple holding the minimum and maximum expected value to predict.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 hidden_units: List[int],
                 non_lin: nn.Module,
                 support_range: Tuple[int, int]):
        super().__init__(obs_shapes, hidden_units, non_lin)

        # build categorical value head
        support_set_size = support_range[1] - support_range[0] + 1
        self.perception_dict["probabilities"] = LinearOutputBlock(
            in_keys="latent", out_keys="probabilities", in_shapes=self.perception_dict["latent"].out_shapes(),
            output_units=support_set_size)

        # compute value as probability weighted sum of supports
        def _to_scalar(x: torch.Tensor) -> torch.Tensor:
            return support_to_scalar(x, support_range=support_range)

        self.perception_dict["value"] = FunctionalBlock(
            in_keys="probabilities", out_keys="value", in_shapes=self.perception_dict["probabilities"].out_shapes(),
            func=_to_scalar
        )

        module_init = make_module_init_normc(std=0.01)
        self.perception_dict["probabilities"].apply(module_init)

        # compile inference model
        self.net = InferenceBlock(in_keys=list(obs_shapes.keys()),
                                  out_keys=["probabilities", "value"],
                                  in_shapes=list(obs_shapes.values()),
                                  perception_blocks=self.perception_dict)

    def forward(self, x):
        """ forward pass. """
        return self.net(x)


class FlattenConcatStateActionValueNet(FlattenConcatBaseNet):
    """Flatten and concatenation state action value model.

    :param obs_shapes: Dictionary mapping of observation names to shapes.
    :param output_shapes: Dictionary mapping of output heads to shapes.
    :param hidden_units: Dictionary mapping of action names to shapes.
    :param non_lin: The non-linearity to apply.
    """

    def __init__(self,
                 obs_shapes: Dict[str, Sequence[int]],
                 output_shapes: Dict[str, Sequence[int]],
                 hidden_units: List[int],
                 non_lin: nn.Module):
        super().__init__(obs_shapes, hidden_units, non_lin)

        # build action head
        module_init = make_module_init_normc(std=0.01)
        for output_key, output_shape in output_shapes.items():
            self.perception_dict[output_key] = LinearOutputBlock(in_keys="latent", out_keys=output_key,
                                                                 in_shapes=self.perception_dict["latent"].out_shapes(),
                                                                 output_units=output_shape[-1])

            self.perception_dict[output_key].apply(module_init)

        # compile inference model
        self.net = InferenceBlock(in_keys=list(obs_shapes.keys()),
                                  out_keys=list(output_shapes.keys()),
                                  in_shapes=list(obs_shapes.values()),
                                  perception_blocks=self.perception_dict)

    def forward(self, x):
        """ forward pass. """
        return self.net(x)
