"""Encapsulation of multiple torch state critics for training in structured environments."""
from abc import abstractmethod
from typing import Mapping, Union, List, Dict

import torch
from gym import spaces
from torch import nn

from maze.core.agent.state_critic import StateCritic
from maze.core.agent.state_critic_input_output import StateCriticStepOutput, StateCriticOutput, StateCriticInput
from maze.core.agent.torch_model import TorchModel
from maze.core.annotations import override
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import StepKeyType
from maze.perception.perception_utils import convert_to_torch, flatten_spaces, stack_and_flatten_spaces


class TorchStateCritic(TorchModel, StateCritic):
    """Encapsulates multiple torch state critics for training in structured environments.

    :param networks: Mapping of value functions (critic) to encapsulate.
    :param obs_spaces_dict: The observation spaces dict of the environment.
    :param device: Device the policy should be located on (cpu or cuda).
    """

    def __init__(self, networks: Mapping[Union[str, int], nn.Module],
                 obs_spaces_dict: Dict[StepKeyType, spaces.Dict], device: str):
        self.networks = networks
        self._num_critics = len(obs_spaces_dict)
        self._obs_spaces_dict = dict(obs_spaces_dict)
        TorchModel.__init__(self, device=device)

    @override(StateCritic)
    def predict_value(self, observation: ObservationType, critic_id: Union[int, str]) -> Dict[str, torch.Tensor]:
        """implementation of :class:`~maze.core.agent.state_critic.StateCritic`
        """
        obs_t = convert_to_torch(observation, device=self._device, cast=None, in_place=False)
        return self.networks[critic_id](obs_t)

    @property
    @abstractmethod
    def num_critics(self) -> int:
        """Returns the number of critic networks.
        :return: Number of critic networks.
        """

    @override(TorchModel)
    def parameters(self) -> List[torch.Tensor]:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        params = []
        for critic in self.networks.values():
            params.extend(list(critic.parameters()))
        return params

    @override(TorchModel)
    def eval(self) -> None:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        for critic in self.networks.values():
            critic.eval()

    @override(TorchModel)
    def train(self) -> None:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        for critic in self.networks.values():
            critic.train()

    @override(TorchModel)
    def to(self, device: str) -> None:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        self._device = device
        for critic in self.networks.values():
            critic.to(device)

    @property
    def device(self) -> str:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        return self._device

    @override(TorchModel)
    def state_dict(self) -> Dict:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        state_dict_critics = dict()

        for key, critic in self.networks.items():
            state_dict_critics[key] = critic.state_dict()

        return dict(critics=state_dict_critics)

    @override(TorchModel)
    def load_state_dict(self, state_dict: Dict) -> None:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        if "critics" in state_dict:
            state_dict_critics = state_dict["critics"]

            for key, critic in self.networks.items():
                assert key in state_dict_critics, f"Could not find state dict for policy ID: {key}"
                critic.load_state_dict(state_dict_critics[key])

    @abstractmethod
    def compute_structured_return(self,
                                  gamma: float,
                                  gae_lambda: float,
                                  rewards: List[torch.Tensor],
                                  values: List[torch.Tensor],
                                  dones: torch.Tensor,
                                  ) -> List[torch.Tensor]:
        """Compute bootstrapped return for the whole structured step (i.e., all sub-steps).

        :param gamma: Discounting factor
        :param gae_lambda: Bias vs variance trade of factor for Generalized Advantage Estimator (GAE)
        :param rewards: List of sub-step rewards, each with shape (n_steps, n_workers)
        :param values: List of sub-step detached values, each with shape (n_steps, n_workers)
        :param dones: Step dones with shape (n_steps, n_workers)
        :return: List of per-time sub-step returns
        """

    def compute_return(self,
                       gamma: float,
                       gae_lambda: float,
                       rewards: torch.Tensor,
                       values: torch.Tensor,
                       dones: torch.Tensor,
                       deltas: torch.Tensor = None,
                       ) -> torch.Tensor:
        """Compute bootstrapped return from rewards and estimated values.

        :param gamma: Discounting factor
        :param gae_lambda: Bias vs variance trade of factor for Generalized Advantage Estimator (GAE)
        :param rewards: Step rewards with shape (n_steps, n_workers)
        :param values: Predicted values with shape (n_steps, n_workers)
        :param dones: Step dones with shape (n_steps, n_workers)
        :param deltas: Predicted value deltas to previous sub-step with shape (n_steps, n_workers)
        :return: Per time step returns.
        """
        assert rewards.shape == values.shape, f'{rewards.shape} vs {values.shape}'
        assert rewards.shape == dones.shape, f'{rewards.shape} vs {dones.shape}'

        # initialize returns
        returns = torch.zeros((rewards.shape[0], rewards.shape[1]), dtype=torch.float32, device=self.device)

        # prepare end-of-episode mask
        mask = (~dones).float()

        # traverse time steps in reverse order
        gae = torch.zeros(rewards.shape[1], dtype=torch.float32, device=self.device)
        for t in reversed(range(0, len(rewards))):

            # bootstrap value function for last entry
            if t == len(rewards) - 1:
                returns[t] = values[t]
                if deltas is not None:
                    returns[t] = values[t] + deltas[t]

            # compute discounted return
            else:

                if gae_lambda != 1.0:
                    delta = rewards[t] + gamma * values[t + 1] * mask[t] - values[t]
                    gae = delta + gamma * gae_lambda * gae
                    returns[t] = gae + values[t]
                else:
                    returns[t] = rewards[t] + gamma * returns[t + 1] * mask[t]

        return returns


class TorchSharedStateCritic(TorchStateCritic):
    """One critic is shared across all sub-steps or actors (default to use for standard gym-style environments).

    In multi-step and multi-agent scenarios, observations from different sub-steps are merged into one.
    Observation keys common across multiple sub-steps are expected to have the same value and are
    present only once in the resulting dictionary.

    Can be instantiated via the
    :class:`~maze.perception.models.critics.shared_state_critic_composer.SharedStateCriticComposer`.
    """

    def __init__(self,
                 networks: Mapping[Union[str, int], nn.Module],
                 obs_spaces_dict: Dict[StepKeyType, spaces.Dict],
                 device: str,
                 stack_observations: bool):
        super().__init__(networks=networks, obs_spaces_dict=obs_spaces_dict, device=device)
        self.stack_observations = stack_observations
        self.network = list(self.networks.values())[0]  # For convenient access to the single network of this critic

    @override(StateCritic)
    def predict_values(self, critic_input: StateCriticInput) -> StateCriticOutput:
        """implementation of :class:`~maze.core.agent.torch_state_critic.TorchStateCritic`
        """
        if self.stack_observations:
            flattened_obs_t = stack_and_flatten_spaces(critic_input.tensor_dict,
                                                       observation_spaces_dict=self._obs_spaces_dict)
        else:
            flattened_obs_t = flatten_spaces(critic_input.tensor_dict)

        value = self.network(flattened_obs_t)["value"][..., 0]
        critic_output = StateCriticOutput()
        for actor_id in critic_input.actor_ids:
            critic_output.append(StateCriticStepOutput(values=value, detached_values=value.detach(), actor_id=actor_id))

        return critic_output

    @override(StateCritic)
    def predict_value(self, observation: ObservationType, critic_id: Union[int, str]) -> torch.Tensor:
        """Predictions depend on previous sub-steps, thus this method is not supported in the delta state critic.
        """
        raise NotImplemented

    @property
    @override(TorchStateCritic)
    def num_critics(self) -> int:
        """There is a single shared critic network."""
        return 1

    @override(TorchStateCritic)
    def compute_structured_return(self,
                                  gamma: float,
                                  gae_lambda: float,
                                  rewards: List[torch.Tensor],
                                  values: List[torch.Tensor],
                                  dones: torch.Tensor,
                                  ) -> List[torch.Tensor]:
        """Compute return based on shared reward (summing the reward across all sub-steps)"""
        # Sum rewards across all sub-steps into a shared reward
        shared_rewards = torch.stack(rewards).sum(dim=0)

        # Note: With shared critic, values are the same for each sub-step --> just take the last one here
        sub_step_return = self.compute_return(gamma=gamma, gae_lambda=gae_lambda,
                                              rewards=shared_rewards, values=values[-1], dones=dones)

        # The same shared return for each sub-step
        return [sub_step_return for _ in values]


class TorchStepStateCritic(TorchStateCritic):
    """Each sub-step or actor gets its individual critic.
    Can be instantiated via the
    :class:`~maze.perception.models.critics.step_state_critic_composer.StepStateCriticComposer`.
    """

    @override(StateCritic)
    def predict_values(self, critic_input: StateCriticInput) -> StateCriticOutput:
        """implementation of :class:`~maze.core.agent.torch_state_critic.TorchStateCritic`
        """
        critic_output = StateCriticOutput()
        for critic_step_input in critic_input:
            value = self.networks[critic_step_input.actor_id.step_key](critic_step_input.tensor_dict)["value"][..., 0]
            critic_output.append(StateCriticStepOutput(values=value, detached_values=value.detach(),
                                                       actor_id=critic_step_input.actor_id))

        return critic_output

    @property
    @override(TorchStateCritic)
    def num_critics(self) -> int:
        """implementation of :class:`~maze.core.agent.torch_state_critic.TorchStateCritic`
        """
        return self._num_critics

    @override(TorchStateCritic)
    def compute_structured_return(self,
                                  gamma: float,
                                  gae_lambda: float,
                                  rewards: List[torch.Tensor],
                                  values: List[torch.Tensor],
                                  dones: torch.Tensor,
                                  ) -> List[torch.Tensor]:
        """Compute returns for each sub-step separately"""
        returns = []
        for substep_rewards, substep_values in zip(rewards, values):
            sub_step_return = self.compute_return(gamma=gamma, gae_lambda=gae_lambda,
                                                  rewards=substep_rewards, values=substep_values, dones=dones)
            returns.append(sub_step_return)

        return returns


class TorchDeltaStateCritic(TorchStateCritic):
    """First sub step gets a regular critic, subsequent sub-steps predict a delta w.r.t. to the previous critic.
    Can be instantiated via the
    :class:`~maze.perception.models.critics.delta_state_critic_composer.DeltaStateCriticComposer`.
    """

    @override(StateCritic)
    def predict_values(self, critic_input: StateCriticInput) -> StateCriticOutput:
        """implementation of :class:`~maze.core.agent.state_critic.StateCritic`"""

        critic_output = StateCriticOutput()
        # predict values for the first state
        key_0 = critic_input[0].actor_id.step_key
        value_0 = self.networks[key_0](critic_input[0].tensor_dict)["value"][..., 0]
        critic_output.append(StateCriticStepOutput(value_0, detached_values=value_0.detach(),
                                                   actor_id=critic_input[0].actor_id))

        for step_critic_input in critic_input.substep_inputs[1:]:
            # compute value 2 as delta of value 1
            prev_values = critic_output.detached_values[-1]
            obs = step_critic_input.tensor_dict.copy()
            obs['prev_value'] = prev_values.unsqueeze(-1)

            value_delta = self.networks[step_critic_input.actor_id.step_key](obs)["value"][..., 0]
            next_values = critic_output.detached_values[-1] + value_delta

            critic_output.append(StateCriticStepOutput(next_values, detached_values=next_values.detach(),
                                                       actor_id=step_critic_input.actor_id))

        return critic_output

    @override(StateCritic)
    def predict_value(self, observation: ObservationType, critic_id: Union[int, str]) -> torch.Tensor:
        """Predictions depend on previous sub-steps, thus this method is not supported in the delta state critic.
        """
        raise NotImplemented

    @property
    @override(TorchStateCritic)
    def num_critics(self) -> int:
        """implementation of :class:`~maze.core.agent.torch_state_critic.TorchStateCritic`
        """
        return self._num_critics

    @override(TorchStateCritic)
    def compute_structured_return(self,
                                  gamma: float,
                                  gae_lambda: float,
                                  rewards: List[torch.Tensor],
                                  values: List[torch.Tensor],
                                  dones: torch.Tensor,
                                  ) -> List[torch.Tensor]:
        """Compute return based on shared reward (summing the reward across all sub-steps)"""
        # Sum rewards across all sub-steps into a shared reward
        shared_rewards = torch.stack(rewards).sum(dim=0)
        sub_step_return = self.compute_return(gamma=gamma, gae_lambda=gae_lambda,
                                              rewards=shared_rewards, values=values[-1], dones=dones)

        # The same shared return for each sub-step
        return [sub_step_return for _ in values]
