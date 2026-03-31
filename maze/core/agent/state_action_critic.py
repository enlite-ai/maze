"""Encapsulates state action critic and queries them for values according to the provided policy ID, observation
and action."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class StateActionCritic(ABC):
    """Structured state action critic class designed to work with structured environments.
    (see :class:`~maze.core.env.structured_env.StructuredEnv`).

    It encapsulates state critic and queries them for values according to the provided policy ID.
    """

    @abstractmethod
    def predict_q_values(
        self,
        observations: dict[str | int, dict[str, torch.Tensor]],
        actions: dict[str | int, dict[str, torch.Tensor]],
        gather_output: bool,
    ) -> dict[str | int, list[torch.Tensor | dict[str, torch.Tensor]]]:
        """Predict the Q value based on the observations and actions.

        :param observations: The observation for the current step.
        :param actions: The action performed at the current step.
        :param gather_output: Specify whether to gather the output in the discrete setting.

        :return: A list of tensors holding the predicted q value for each critic.
        """
