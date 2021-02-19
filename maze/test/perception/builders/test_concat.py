""" Contains unit tests for concat model builder """
import numpy as np
from gym import spaces

from maze.perception.blocks.inference import InferenceBlock
from maze.perception.builders.concat import ConcatModelBuilder


def build_dict_obs_space():
    """build dictionary observation space"""
    space_dict = dict()
    space_dict["features"] = spaces.Box(low=0.0, high=1.0, shape=(24,), dtype=np.float64)
    space_dict["image"] = spaces.Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float64)

    return spaces.Dict(space_dict)


def test_concat_model_builder() -> None:
    """ main
    """

    # --- map observations to perception observation modalities (can go to config) ---
    obs_modalities = {"features": "feature",
                      "image": "image"}

    # set modality config
    modality_config = dict()
    modality_config["feature"] = {"block_type": "maze.perception.blocks.DenseBlock",
                                  "block_params": {"hidden_units": [64, 64],
                                                   "non_lin": "torch.nn.ReLU"}}
    modality_config["image"] = {"block_type": "maze.perception.blocks.VGGConvolutionDenseBlock",
                                "block_params": {"hidden_channels": [8, 16],
                                                 "hidden_units": [64, 64],
                                                 "non_lin": "torch.nn.ReLU"}}

    for rnn_steps in [0, 2]:

        # prepare observation space
        dict_space = build_dict_obs_space()

        if rnn_steps > 1:
            dict_space = model_builder.to_recurrent_gym_space(dict_space, rnn_steps=rnn_steps)

            modality_config["hidden"] = {}
            modality_config["recurrence"] = {"block_type": "maze.perception.blocks.LSTMLastStepBlock",
                                             "block_params": {"hidden_size": 32,
                                                              "num_layers": 1,
                                                              "bidirectional": False,
                                                              "non_lin": "torch.nn.ReLU"}}
        else:
            modality_config["hidden"] = {"block_type": "maze.perception.blocks.DenseBlock",
                                         "block_params": {"hidden_units": [64, 64],
                                                          "non_lin": "torch.nn.ReLU"}}
            modality_config["recurrence"] = {}

        # build models
        model_builder = ConcatModelBuilder(modality_config, obs_modalities)
        model = model_builder.from_observation_space(observation_space=dict_space)
        assert isinstance(model, InferenceBlock)
        assert "latent" in model.out_keys
        assert "features" in model.in_keys
        assert "image" in model.in_keys
