"""Encapsulation of multiple torch state critics for training in structured environments."""
from abc import abstractmethod
from typing import Mapping, Union, List, Dict, Tuple

import torch
from torch import nn

from maze.core.agent.state_critic import StateCritic
from maze.core.agent.torch_model import TorchModel
from maze.core.annotations import override
from maze.core.env.observation_conversion import ObservationType
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.perception.perception_utils import convert_to_torch, flatten_spaces


class TorchStateCritic(TorchModel, StateCritic):
    """Encapsulates multiple torch state critics for training in structured environments.

    :param networks: Mapping of value functions (critic) to encapsulate.
    :param num_policies: The number of corresponding policies.
    :param device: Device the policy should be located on (cpu or cuda)
    """

    def __init__(self, networks: Mapping[Union[str, int], nn.Module], num_policies: int, device: str):
        self.networks = networks
        self.num_policies = num_policies
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

    def bootstrap_returns(self, record: StructuredSpacesRecord, gamma: float, gae_lambda: float,) \
            -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Bootstrap returns using the value function.

        Useful for example to implement PPO or A2C.
        
        :param observations: Sub-step observations as tensor dictionary.
        :param rews: Array holding the per step rewards.
        :param dones: Array indicating if a step is a done step.
        :param gamma: Discounting factor
        :param gae_lambda: Bias vs variance trade of factor for Generalized Advantage Estimator (GAE)
        :return: Tuple containing the computed returns, the predicted values and the detached predicted values.
        """

        # predict state values
        values, detached_values = self.predict_values(record)

        # TODO: Currently only reward & done from the last sub-step is considered --> consider other sub-steps as well
        rews = record.rewards[-1]
        dones = record.dones[-1]

        # reshape values to match rewards
        values = [torch.reshape(v, rews.shape) for v in values]
        detached_values = [torch.reshape(dv, rews.shape) for dv in detached_values]

        # compute returns
        sub_step_return = self.compute_return(gamma=gamma, gae_lambda=gae_lambda,
                                              rewards=rews, values=detached_values[-1], dones=dones)

        returns = [sub_step_return for _ in values]
        return returns, values, detached_values

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
        assert rewards.shape == values.shape
        assert rewards.shape == dones.shape

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
    Can be instantiated via the
    :class:`~maze.perception.models.critics.shared_state_critic_composer.SharedStateCriticComposer`.
    """

    def __init__(self, networks: Mapping[Union[str, int], nn.Module], num_policies: int, device: str):
        super().__init__(networks=networks, num_policies=num_policies, device=device)

        self.network = list(self.networks.values())[0]  # For convenient access to the single network of this critic

    @override(StateCritic)
    def predict_values(self, record: StructuredSpacesRecord) -> \
            Tuple[Dict[Union[str, int], torch.Tensor], Dict[Union[str, int], torch.Tensor]]:
        """Predict the shared values and repeat them for each sub-step."""
        flattened_obs_t = flatten_spaces(record.observations)
        value = self.network(flattened_obs_t)["value"][..., 0]

        values = [value for _ in record.substep_records]
        detached_values = [value.detach() for _ in record.substep_records]

        return values, detached_values

    @property
    @override(TorchStateCritic)
    def num_critics(self) -> int:
        """There is a single shared critic network."""
        return 1


class TorchStepStateCritic(TorchStateCritic):
    """Each sub-step or actor gets its individual critic.
    Can be instantiated via the
    :class:`~maze.perception.models.critics.step_state_critic_composer.StepStateCriticComposer`.
    """

    @override(StateCritic)
    def predict_values(self, observations: Dict[Union[str, int], Dict[str, torch.Tensor]]) -> \
            Tuple[Dict[Union[str, int], torch.Tensor], Dict[Union[str, int], torch.Tensor]]:
        """implementation of :class:`~maze.core.agent.state_critic.StateCritic`
        """
        observations = convert_to_torch(observations, device=self._device, cast=None, in_place=False)

        values, detached_values = dict(), dict()
        for step_id in observations.keys():
            values[step_id] = self.networks[step_id](observations[step_id])["value"][..., 0]
            detached_values[step_id] = values[step_id].detach()

        return values, detached_values

    @property
    @override(TorchStateCritic)
    def num_critics(self) -> int:
        """implementation of :class:`~maze.core.agent.torch_state_critic.TorchStateCritic`
        """
        return self.num_policies


class TorchDeltaStateCritic(TorchStateCritic):
    """First sub step gets a regular critic, subsequent sub-steps predict a delta w.r.t. to the previous critic.
    Can be instantiated via the
    :class:`~maze.perception.models.critics.delta_state_critic_composer.DeltaStateCriticComposer`.
    """

    @override(StateCritic)
    def predict_values(self, observations: Dict[Union[str, int], Dict[str, torch.Tensor]]) -> \
            Tuple[Dict[Union[str, int], torch.Tensor], Dict[Union[str, int], torch.Tensor]]:
        """implementation of :class:`~maze.core.agent.state_critic.StateCritic`
        """
        observations = convert_to_torch(observations, device=self._device, cast=None, in_place=False)

        sub_step_keys = list(observations.keys())

        # predict values for first state
        key_0 = sub_step_keys[0]
        values = {key_0: self.networks[key_0](observations[key_0])["value"][..., 0]}
        detached_values = {key_0: values[0].detach()}

        for i, step_id in enumerate(sub_step_keys[1:], start=1):
            # compute value 2 as delta of value 1
            prev_values = detached_values[sub_step_keys[i - 1]]
            observations[step_id].update({'prev_value': prev_values.unsqueeze(-1)})
            value_delta = self.networks[step_id](observations[step_id])["value"][..., 0]
            next_values = detached_values[sub_step_keys[i - 1]] + value_delta

            values[step_id] = next_values
            detached_values[step_id] = values[step_id].detach()

        return values, detached_values

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
        return self.num_policies
