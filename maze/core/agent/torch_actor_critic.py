"""Encapsulation of multiple torch policies for training and rollouts in structured environments."""
from typing import List, Dict, Union

import torch

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_action_critic import TorchStateActionCritic
from maze.core.agent.torch_state_critic import TorchStateCritic
from maze.core.agent.torch_model import TorchModel
from maze.core.annotations import override


class TorchActorCritic(TorchModel):
    """Encapsulates a structured torch policy and critic
    for training actor-critic algorithms in structured environments.

    :param policy: A structured torch policy for training in structured environments.
    :param critic: A structured torch critic for training in structured environments.
    :param device: Device the model (networks) should be located on (cpu or cuda)
    """

    def __init__(self,
                 policy: TorchPolicy,
                 critic: Union[TorchStateCritic, TorchStateActionCritic],
                 device: str):
        # check if appropriate models are provided
        assert critic is not None and isinstance(critic, (TorchStateCritic,
                                                          TorchStateActionCritic)), \
            "Make sure to provide an appropriate critic when training with actor-critic models!"
        assert policy is not None and isinstance(policy, TorchPolicy), \
            "Make sure to provide an appropriate policy when training with actor-critic models!"

        self.policy = policy
        self.critic = critic

        TorchModel.__init__(self, device=device)

    @override(TorchModel)
    def parameters(self) -> List[torch.Tensor]:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        return self.policy.parameters() + self.critic.parameters()

    @override(TorchModel)
    def eval(self) -> None:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        self.policy.eval()
        self.critic.eval()

    @override(TorchModel)
    def train(self) -> None:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        self.policy.train()
        self.critic.train()

    @override(TorchModel)
    def to(self, device: str):
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        self._device = device
        self.policy.to(device)
        self.critic.to(device)

    @property
    def device(self) -> str:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        return self._device

    @override(TorchModel)
    def state_dict(self) -> Dict:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        state_dict = dict()
        state_dict.update(self.policy.state_dict())
        state_dict.update(self.critic.state_dict())
        return state_dict

    @override(TorchModel)
    def load_state_dict(self, state_dict: Dict) -> None:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        self.policy.load_state_dict(state_dict)
        self.critic.load_state_dict(state_dict)
