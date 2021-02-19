"""Contains perception unit tests."""
from maze.perception.models.built_in.flatten_concat import FlattenConcatPolicyNet

from gym import spaces
import torch
from maze.perception.perception_utils import convert_to_torch


def test_model():
    """ unit tests """

    obs_space = spaces.Dict(spaces={"obs_1": spaces.Box(low=0, high=1, shape=(20, 10)),
                                    "obs_2": spaces.Box(low=0, high=1, shape=(20,))})

    obs_shapes = {"obs_1": (20, 10), "obs_2": (20, )}
    action_logits_shapes = {"act_1": (5, ), "act_2": (10, )}

    policy_net = FlattenConcatPolicyNet(obs_shapes=obs_shapes,
                                        action_logits_shapes=action_logits_shapes,
                                        hidden_units=[32, 32],
                                        non_lin=torch.nn.ReLU)

    obs_tensor = convert_to_torch(obs_space.sample(), cast=None, in_place=True, device="cpu")
    action_logits_dict = policy_net.forward(obs_tensor)
    assert "act_1" in action_logits_dict
    assert "act_2" in action_logits_dict
    assert action_logits_dict["act_1"].shape == (5,)
    assert action_logits_dict["act_2"].shape == (10,)
