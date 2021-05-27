"""Loss function for behavioral cloning."""
from dataclasses import dataclass
from typing import Dict, Union, Any

import gym
import torch
import torch.nn as nn
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.env.structured_env import ActorID
from maze.train.trainers.imitation.imitation_events import ImitationEvents


@dataclass
class BCLoss:
    """Loss function for behavioral cloning."""

    action_spaces_dict: Dict[Union[int, str], gym.spaces.Dict]
    """Action space we are training on (used to determine appropriate loss functions)"""

    entropy_coef: float
    """Weight of entropy loss"""

    loss_discrete: nn.Module = nn.CrossEntropyLoss()
    """Loss function used for discrete (categorical) spaces"""

    loss_box: nn.Module = nn.MSELoss()
    """Loss function used for box (continuous) spaces"""

    loss_multi_binary: nn.Module = nn.functional.binary_cross_entropy_with_logits
    """Loss function used for multi-binary spaces"""

    def calculate_loss(self,
                       policy: TorchPolicy,
                       observation_dict: Dict[Union[int, str], Any],
                       action_dict: Dict[Union[int, str], Any],
                       events: ImitationEvents
                       ) -> torch.Tensor:
        """Calculate and return the training loss for one step (= multiple sub-steps in structured scenarios).

        :param policy: Structured policy to evaluate
        :param observation_dict: Dictionary with observations identified by substep ID
        :param action_dict: Dictionary with actions identified by substep ID
        :param events:
        :return: Total loss
        """
        losses = []

        # Iterate over all substeps
        for substep_key, observation in observation_dict.items():
            target = action_dict[substep_key]
            logits = policy.compute_logits_dict(observation, actor_id=ActorID(substep_key, 0))
            substep_losses = self._get_substep_loss(substep_key, logits, target, self.action_spaces_dict[substep_key],
                                                    events=events)
            losses.append(substep_losses)

            # Compute and report policy entropy
            entropy = policy.logits_dict_to_distribution(logits).entropy().mean()
            events.policy_entropy(step_id=substep_key, value=entropy.item())
            if self.entropy_coef > 0:
                losses.append(-self.entropy_coef * entropy)

        return sum(losses)

    def _get_substep_loss(self, step_id: str,
                          logits_dict: Dict[str, torch.Tensor],
                          target_dict: Dict[str, torch.Tensor],
                          action_spaces_dict: gym.spaces.Dict,
                          events: ImitationEvents) -> torch.Tensor:
        """Iterate over the action space of a given policy and calculate the loss based on the types of the
        subspaces.

        :param step_id: The step id to compute the loss for
        :param logits_dict: Logits dict output by the policy for a given substep
        :param target_dict: Target action for the given substep
        :param action_spaces_dict: Dict action space for the given substep
        :return: Total loss for this substep
        """
        losses = []
        for subspace_id, logits in logits_dict.items():
            target = target_dict[subspace_id]
            action_space = action_spaces_dict[subspace_id]

            # Categorical (discrete spaces)
            if isinstance(action_space, gym.spaces.Discrete):
                if logits.dim() == target.dim():
                    logits = logits.unsqueeze(0)
                losses.append(self.loss_discrete(logits, target))
                events.discrete_accuracy(
                    step_id=step_id, subspace_name=subspace_id,
                    value=torch.eq(logits.argmax(dim=-1), target).float().mean().item())

            # Multi-binary (multi-binary spaces)
            elif isinstance(action_space, gym.spaces.MultiBinary):
                losses.append(self.loss_multi_binary(logits, target))
                events.multi_binary_accuracy(step_id=step_id, subspace_name=subspace_id,
                                             value=torch.eq(logits, target).float().mean().item())

            # Continuous (box spaces)
            elif isinstance(action_space, gym.spaces.Box):
                pred, _ = torch.chunk(logits, chunks=2, dim=-1)
                losses.append(self.loss_box(pred.float(), target.float()))
                events.box_mean_abs_deviation(step_id=step_id, subspace_name=subspace_id,
                                              value=target.sub(pred).abs().mean().item())

            else:
                raise NotImplementedError("Only Discrete, Box, and MultiBinary action spaces are supported.")

        loss = sum(losses)
        events.policy_loss(step_id=step_id, value=loss.item())
        return loss
