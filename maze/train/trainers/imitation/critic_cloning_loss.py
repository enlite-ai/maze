"""Loss function for behavioral cloning."""
from dataclasses import dataclass
from typing import Dict, Union, Any

import torch
import torch.nn as nn
from maze.core.agent.torch_state_critic import TorchStateCritic
from maze.train.trainers.imitation.imitation_events import CriticImitationEvents


@dataclass
class StateCriticCloningLoss:
    """Loss function for critic cloning."""

    mse_loss: nn.Module = nn.MSELoss()
    """Loss function used for continuous value approximation."""

    def calculate_loss(self,
                       critic: TorchStateCritic,
                       observation_dict: Dict[Union[int, str], Any],
                       step_returns: Dict[Union[int, str], Any],
                       events: CriticImitationEvents
                       ) -> torch.Tensor:
        """Calculate and return the training loss for one step (= multiple sub-steps in structured scenarios).

        :param critic: Structured critic to evaluate.
        :param observation_dict: Dictionary with observations identified by substep ID
        :param step_returns: The step returns to approximate.
        :param events: Critic imitation events.
        :return: Total loss.
        """
        losses = []

        # Iterate over all substeps
        for policy_id, observation in observation_dict.items():
            # prediction target
            step_return = step_returns[policy_id]
            # prediction
            value = critic.predict_value(observation=observation, critic_id=policy_id)["value"]
            # prediction loss
            substep_loss = self._get_substep_loss(policy_id, value, step_return, events=events)
            losses.append(substep_loss)

        return sum(losses)

    def _get_substep_loss(self, step_id: str, value: torch.Tensor, step_return: torch.Tensor,
                          events: CriticImitationEvents) -> torch.Tensor:
        """Calculate the critic cloning (value function) loss.

        :param step_id: The step id to compute the loss for.
        :param value: The model predicted value.
        :param step_return: The target value (discounted step return).
        :param events: Critic imitation events.
        :return: Total loss for this sub step.
        """
        value = value[:, 0]
        assert value.shape == step_return.shape
        loss = self.mse_loss(value.float(), step_return.float())

        events.value(step_id=step_id, value=value.mean().item())
        events.mean_abs_deviation(step_id=step_id, value=step_return.sub(value).abs().mean().item())
        events.critic_loss(step_id=step_id, value=loss.item())

        return loss
