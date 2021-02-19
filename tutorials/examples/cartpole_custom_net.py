"""Example custom cartpole net."""

from torch import nn

from maze.perception.blocks.feed_forward.dense import DenseBlock
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.weight_init import make_module_init_normc


class PolicyNet(nn.Module):
    """Simple feed forward policy network.
    """

    def __init__(self, obs_shapes, action_logits_shapes, non_lin=nn.Tanh):
        super().__init__()

        # build perception part
        self.perception_network = DenseBlock(in_keys="observation", out_keys="latent",
                                             in_shapes=obs_shapes['observation'],
                                             hidden_units=[32, 32], non_lin=non_lin)

        module_init = make_module_init_normc(std=1.0)
        self.perception_network.apply(module_init)

        # build action head
        self.action_head = LinearOutputBlock(in_keys="latent", out_keys="action",
                                             in_shapes=self.perception_network.out_shapes(),
                                             output_units=action_logits_shapes['action'][-1])

        module_init = make_module_init_normc(std=0.01)
        self.action_head.apply(module_init)

        # compile inference model
        self.net = InferenceBlock(in_keys="observation", out_keys="action", in_shapes=list(obs_shapes.values()),
                                  perception_blocks={"latent": self.perception_network,
                                                     "action": self.action_head})

    def forward(self, x):
        """ forward pass. """
        return self.net(x)


class ValueNet(nn.Module):
    """Simple feed forward value network.
    """

    def __init__(self, obs_shapes, non_lin=nn.Tanh):
        super().__init__()

        # build perception part
        self.perception_network = DenseBlock(in_keys="observation", out_keys="latent",
                                             in_shapes=obs_shapes['observation'],
                                             hidden_units=[32, 32], non_lin=non_lin)

        module_init = make_module_init_normc(std=1.0)
        self.perception_network.apply(module_init)

        # build action head
        self.value_head = LinearOutputBlock(in_keys="latent", out_keys="value",
                                            in_shapes=self.perception_network.out_shapes(),
                                            output_units=1)

        module_init = make_module_init_normc(std=0.01)
        self.value_head.apply(module_init)

        # compile inference model
        self.net = InferenceBlock(in_keys="observation", out_keys="value", in_shapes=list(obs_shapes.values()),
                                  perception_blocks={"latent": self.perception_network,
                                                     "value": self.value_head})

    def forward(self, x):
        """ forward pass. """
        return self.net(x)
