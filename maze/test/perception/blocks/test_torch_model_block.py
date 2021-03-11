""" Unit tests for TorchModelBlocks. """
from typing import Dict

import torch
import torch.nn as nn
from maze.perception.blocks.general.torch_model_block import TorchModelBlock
from maze.test.perception.perception_test_utils import build_multi_input_dict


class CustomPytorchModel(nn.Module):
    """ Dummy PyTorch model """
    def __init__(self):
        super(CustomPytorchModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1),
        )
        self.dense = nn.Sequential(
            nn.Linear(in_features=16, out_features=32)
        )

    def forward(self, in_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ Forward method """
        out_dict = dict()

        out_dict['out_key_0'] = self.cnn(in_dict['in_key_0'])
        out_dict['out_key_1'] = self.dense(in_dict['in_key_1'])

        return out_dict


def test_torch_model_block():
    """ perception test """
    with torch.no_grad():
        custom_model = CustomPytorchModel()
        torch_model_block = TorchModelBlock(
            in_keys=['in_key_0', 'in_key_1'], out_keys=['out_key_0', 'out_key_1'],
            in_shapes=[[3, 5, 5], [16]],
            in_num_dims=[4, 2], out_num_dims=[4, 2], net=custom_model)

        in_tensor_dict = build_multi_input_dict(dims=[[3, 5, 5], [16]])
        out_dict = torch_model_block(in_tensor_dict)
        assert out_dict['out_key_0'].numpy().shape == (4, 5, 5)
        assert out_dict['out_key_1'].numpy().shape == (32,)

        in_tensor_dict = build_multi_input_dict(dims=[[8, 3, 5, 5], [8, 16]])
        out_dict = torch_model_block(in_tensor_dict)
        assert out_dict['out_key_0'].numpy().shape == (8, 4, 5, 5)
        assert out_dict['out_key_1'].numpy().shape == (8, 32)
