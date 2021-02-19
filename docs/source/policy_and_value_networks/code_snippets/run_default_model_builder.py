"""Builds an inference graph visualization from a given model config."""
import gym
import matplotlib.pyplot as plt
import numpy as np
import yaml

from maze.perception.blocks.inference import InferenceGraph
from maze.perception.models.critics import SharedStateCriticComposer
from maze.perception.models.template_model_composer import TemplateModelComposer


def build_model_visualization(config_file: str):
    """ helper function """

    if config_file == "cartpole_concat_model_builder.yaml":
        # init spaces
        observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        })
        action_space = gym.spaces.Dict({
            "action": gym.spaces.Discrete(n=2),
        })
    else:
        if config_file == "ff_concat_model_builder.yaml":
            in_dims = [(16,), (3, 64, 64)]
        elif config_file == "rnn_concat_model_builder.yaml":
            in_dims = [(8, 16,), (8, 3, 64, 64)]

        # init spaces
        observation_space = gym.spaces.Dict({
            "observation_inventory": gym.spaces.Box(low=0, high=1, shape=in_dims[0], dtype=np.float32),
            "observation_screen": gym.spaces.Box(low=0, high=1, shape=in_dims[1], dtype=np.float32)
        })
        action_space = gym.spaces.Dict({
            "action_move": gym.spaces.Discrete(n=4),
            "action_use": gym.spaces.MultiBinary(n=16)
        })

    # load config
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # initialize default model builder
    default_builder = TemplateModelComposer(
        distribution_mapper_config=config["distribution_mapper_config"],
        model_builder=config["model_builder"],
        critic={'type': SharedStateCriticComposer},
        observation_spaces_dict={0: observation_space},
        action_spaces_dict={0: action_space})

    # test default policy gradient actor
    policy_net = default_builder.template_policy_net(observation_space=observation_space, action_space=action_space)

    graph = InferenceGraph(inference_block=policy_net)
    graph.show(name='Policy Network', block_execution=False)

    # test standalone critic
    value_net = default_builder.template_value_net(observation_space=observation_space)

    graph = InferenceGraph(inference_block=value_net)
    graph.show(name='Value Network', block_execution=False)

    plt.show(block=True)


if __name__ == "__main__":
    """ main """
    build_model_visualization(config_file="cartpole_concat_model_builder.yaml")
