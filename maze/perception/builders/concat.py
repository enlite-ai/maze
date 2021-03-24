""" Contains a concatenation model builder. """
from typing import Dict, Union, Any

from gym import spaces

from maze.core.annotations import override
from maze.core.utils.factory import Factory
from maze.perception.blocks import PerceptionBlock
from maze.perception.blocks.general.concat import ConcatenationBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.builders.base import BaseModelBuilder
from maze.utils.bcolors import BColors


class ConcatModelBuilderKeys:
    """Reserved Keys for Model Builders"""

    RECURRENCE: str = "recurrence"
    """ This key is reserved for recurrent blocks. """

    HIDDEN: str = "hidden"
    """ This key is reserved for hidden layers. """

    CONCAT: str = "concat"
    """ This key is reserved for the output of the concatenation block. """

    LATENT: str = "latent"
    """ This key is reserved for the final output of the feature learner. """


class ConcatModelBuilder(BaseModelBuilder):
    """A model builder that first processes individual observations, concatenates the resulting latent spaces and then
    processes this concatenated output to action and value outputs.

    Each input observation is first processed with the specified perception block.
    The required feature dimensionality after this step is 1D!
    In a next step the latent representations of the previous step are concatenated along the last dimension and
    once more processed with a :class:`~maze.perception.blocks.feed_forward.dense.DenseBlock`.

    :param: modality_config: dictionary mapping perception modalities to blocks and block config parameters.
    :param observation_modality_mapping: A mapping of observation keys to perception modalities.
    """

    def __init__(self, modality_config: Dict[str, Union[str, Dict[str, Any]]],
                 observation_modality_mapping: Dict[str, str]):
        self._check_modality_config(modality_config)
        super().__init__(modality_config, observation_modality_mapping)

        # map modalities to blocks
        self.obs_to_block: Dict[str, PerceptionBlock] = dict()
        self.block_params = dict()
        for modality, config in self.modality_config.items():
            if config != {}:
                self.obs_to_block[modality] = Factory(PerceptionBlock).type_from_name(config["block_type"])
                self.block_params[modality] = config["block_params"]

    @override(BaseModelBuilder)
    def from_observation_space(self, observation_space: spaces.Dict) -> InferenceBlock:
        """implementation of :class:`~maze.perception.builders.base.BaseModelBuilder` interface
        """

        # get a sample observation
        sample = observation_space.sample()

        # init perception dict
        perception_dict = dict()
        in_keys = list()

        # --- iterate and process observations ---
        for obs_key in observation_space.spaces.keys():
            if obs_key not in self.observation_modality_mapping:
                BColors.print_colored(
                    f'ConcatModelBuilder: The observation \'{obs_key}\' could not be found in the '
                    f'model_builder.observation_modality_mapping and wont be considered as an input to the network.',
                    BColors.WARNING)
                continue
            in_keys.append(obs_key)
            modality = self.observation_modality_mapping[obs_key]
            block_type = self.obs_to_block[modality]

            # compile network block
            params = self.block_params[modality]
            net = block_type(in_keys=obs_key, out_keys=f"{obs_key}_{block_type.__name__}",
                             in_shapes=sample[obs_key].shape, **params)
            perception_dict[f"{obs_key}_{block_type.__name__}"] = net

        # --- merge latent space observations ---
        out_key = ConcatModelBuilderKeys.CONCAT
        if ConcatModelBuilderKeys.HIDDEN not in self.obs_to_block \
                and ConcatModelBuilderKeys.RECURRENCE not in self.obs_to_block:
            out_key = ConcatModelBuilderKeys.LATENT

        latent_keys = list(perception_dict.keys())
        latent_shapes = [net.out_shapes()[0] for net in perception_dict.values()]
        net = ConcatenationBlock(in_keys=latent_keys, out_keys=out_key,
                                 in_shapes=latent_shapes, concat_dim=-1)
        perception_dict[out_key] = net

        # --- process with presets ---
        if ConcatModelBuilderKeys.HIDDEN in self.obs_to_block:
            in_key = out_key
            out_key = ConcatModelBuilderKeys.HIDDEN
            if ConcatModelBuilderKeys.RECURRENCE not in self.obs_to_block:
                out_key = ConcatModelBuilderKeys.LATENT

            block_type = self.obs_to_block[ConcatModelBuilderKeys.HIDDEN]
            net = block_type(in_keys=in_key, out_keys=out_key,
                             in_shapes=perception_dict[in_key].out_shapes(),
                             **self.block_params[ConcatModelBuilderKeys.HIDDEN])
            perception_dict[out_key] = net

        if ConcatModelBuilderKeys.RECURRENCE in self.obs_to_block:
            in_key = out_key
            out_key = ConcatModelBuilderKeys.LATENT

            block_type = self.obs_to_block[ConcatModelBuilderKeys.RECURRENCE]
            net = block_type(in_keys=in_key, out_keys=out_key,
                             in_shapes=perception_dict[in_key].out_shapes(),
                             **self.block_params[ConcatModelBuilderKeys.RECURRENCE])
            perception_dict[out_key] = net

        # compile inference block
        in_shapes = [sample[obs_key].shape for obs_key in in_keys]
        net = InferenceBlock(in_keys=in_keys, out_keys=ConcatModelBuilderKeys.LATENT,
                             in_shapes=in_shapes, perception_blocks=perception_dict)

        return net

    def _check_modality_config(self, modality_config: Dict):

        assert ConcatModelBuilderKeys.RECURRENCE in modality_config, \
            f"make sure to specify a block for {ConcatModelBuilderKeys.RECURRENCE}!"
        assert ConcatModelBuilderKeys.HIDDEN in modality_config, \
            f"make sure to specify a block for {str(ConcatModelBuilderKeys.HIDDEN)}!"

        for key, value in modality_config.items():
            if value != {}:
                assert list(value.keys()) == ["block_type", "block_params"], \
                    f"{self.__class__.__name__} requires 'block_type' and 'block_params' as arguments" \
                    f" for modality '{key}'."
