"""Encapsulation of multiple torch state action critics for training in structured environments."""
import copy
import itertools
from abc import abstractmethod
from typing import Mapping, Union, List, Dict

import torch
from gym import spaces
from torch import nn
from torch.distributions.utils import logits_to_probs

from maze.core.agent.state_action_critic import StateActionCritic
from maze.core.agent.torch_model import TorchModel
from maze.core.annotations import override
from maze.core.wrappers.observation_preprocessing.preprocessors.one_hot import OneHotPreProcessor
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.perception_utils import flatten_spaces, convert_to_numpy, convert_to_torch
from maze.perception.weight_init import make_module_init_normc
from maze.utils.bcolors import BColors


class TorchStateActionCritic(TorchModel, StateActionCritic):
    """Encapsulates multiple torch state action critics for training in structured environments.

    :param networks: Mapping of value functions (critic) to encapsulate.
    :param num_policies: The number of corresponding policies.
    :param device: Device the policy should be located on (cpu or cuda)
    :param only_discrete_spaces: A dict specifying if the action spaces w.r.t. the step only hold discrete action
                                 spaces.
    """

    target_key = '_target'

    def __init__(self, networks: Mapping[Union[str, int], nn.Module], num_policies: int, device: str,
                 only_discrete_spaces: Dict[Union[str, int], bool],
                 action_spaces_dict: Dict[Union[str, int], spaces.Dict]):

        # TODO: make this a hyperparameter
        self._num_critics_per_step = 2
        self.step_critic_keys = list(networks.keys())

        # Add target networks
        networks = dict(networks)
        networks.update({(old_name, self.target_key): copy.deepcopy(critic) for old_name, critic in networks.items()})

        # Add multi q networks
        self.critic_key_mapping = dict()
        new_networks = dict()
        for old_name, step_critic in networks.items():
            tmp_critics = {(old_name, idx): copy.deepcopy(step_critic) for idx in range(self._num_critics_per_step)}
            self.critic_key_mapping[old_name] = list(tmp_critics.keys())
            new_networks.update(tmp_critics)
        self.networks = new_networks

        self.num_policies = num_policies
        self.only_discrete_spaces = only_discrete_spaces
        self._preprocessors = dict()
        for step_key, only_discrete in self.only_discrete_spaces.items():
            if not only_discrete:
                discrete_spaces_keys = [key for key, value in action_spaces_dict[step_key].spaces.items()
                                        if isinstance(value, spaces.Discrete)]
                for action_key in discrete_spaces_keys:
                    for critic_key in [step_key, (step_key, self.target_key)]:
                        if critic_key not in self._preprocessors:
                            self._preprocessors[critic_key] = dict()
                        self._preprocessors[critic_key][action_key] = OneHotPreProcessor(
                            action_spaces_dict[step_key][action_key])

        TorchModel.__init__(self, device=device)
        self.re_init_networks()

    def compute_state_action_value_step(self, observation: Dict[str, torch.Tensor], action: Dict[str, torch.Tensor],
                                        critic_id: Union[str, int, tuple]) -> List[torch.Tensor]:
        """Predict the value with specified step_key, step_observation and action.

        :param observation: The observation for the current step.
        :param action: The action performed at the current step.
        :param critic_id: The current step key of the multi-step env.

        :return: A list of tensors holding the predicted q value for each critic.
        """

        if critic_id in self._preprocessors:
            for action_key, action_value in action.items():
                if action_key in self._preprocessors[critic_id]:
                    np_action = convert_to_numpy(action_value, cast=None, in_place=False)
                    processed_action = self._preprocessors[critic_id][action_key].process(np_action)
                    action[action_key] = convert_to_torch(processed_action, device=self.device, cast=torch.float32,
                                                          in_place=False)
        observation.update(action)

        sub_critic_keys = self.critic_key_mapping[critic_id]
        q_values = []
        for idx, sub_critic_key in enumerate(sub_critic_keys):
            q_values.append(self.networks[sub_critic_key](observation)['q_value'].squeeze(-1))

        return q_values

    def compute_state_action_values_step(self, observation: Dict[str, torch.Tensor],
                                         critic_id: Union[str, int, tuple]) -> List[Dict[str, torch.Tensor]]:
        """Predict the value with specified step_key, step_observation and action for discrete actions only.

        :param observation: The observation for the current step.
        :param critic_id: The current step key of the multi-step env.

        :return: A list of dicts holding the predicted q value for each action w.r.t. to the critic.
        """

        sub_critic_keys = self.critic_key_mapping[critic_id]
        q_values = []
        for idx, sub_critic_key in enumerate(sub_critic_keys):
            net_out = self.networks[sub_critic_key](observation)
            for value in net_out.values():
                value.unsqueeze(-1)
            q_values.append(net_out)

        return q_values

    @override(StateActionCritic)
    @abstractmethod
    def predict_q_values(self, observations: Dict[Union[str, int], Dict[str, torch.Tensor]],
                         actions: Dict[Union[str, int], Dict[str, torch.Tensor]], gather_output: bool) -> \
            Dict[Union[str, int], List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        """implementation of :class:`~maze.core.agent.state_action_critic.StateActionCritic`
        """

    @abstractmethod
    def predict_next_q_values(self, next_observations: Dict[Union[str, int], Dict[str, torch.Tensor]],
                              next_actions: Dict[Union[str, int], Dict[str, torch.Tensor]],
                              next_actions_logits: Dict[Union[str, int], Dict[str, torch.Tensor]],
                              next_actions_log_probs: Dict[Union[str, int], Dict[str, torch.Tensor]],
                              alpha: Dict[Union[str, int], torch.Tensor]) \
            -> Dict[Union[str, int], Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Predict the target q value for the next step.  :math:`V (st) := E_{at∼π}[Q(st, at) − α log(π(at |st))]`.

        :param next_observations: The next observations.
        :param next_actions: The next actions sampled from the policy.
        :param next_actions_logits: The logits of the next actions (only relevantt for the discrete case).
        :param next_actions_log_probs: The log probabilities of the actions.
        :param alpha: The alpha or entropy coefficient for each step.

        :return: A dict w.r.t. the step holding tensors representing the predicted next q value
        """

    @property
    @abstractmethod
    def num_critics(self) -> int:
        """Returns the number of critic networks.
        :return: Number of critic networks.
        """

    def per_critic_parameters(self) -> List[List[torch.Tensor]]:
        """Retrieve all trainable critic parameters (to be assigned to optimizers).
        :return: List of lists holding all parameters for the base critic corresponding to number of critic per step.
        """
        params = []
        for critic_num in range(self._num_critics_per_step):
            params.append([])
            for step_key in self.step_critic_keys:
                params[critic_num].append(self.networks[self.critic_key_mapping[step_key][critic_num]].parameters())
        for critic_num in range(self._num_critics_per_step):
            params[critic_num] = itertools.chain(*params[critic_num])
        return params

    def update_target_weights(self, tau: float) -> None:
        """Preform a soft or hard update depending on the tau value chosen. tau==1 results in a hard update

        :param tau: Parameter weighting the soft update of the target network.
        """
        for step_key in self.step_critic_keys:
            for critic_key, target_critic_key in zip(self.critic_key_mapping[step_key],
                                                     self.critic_key_mapping[(step_key, self.target_key)]):
                # Copy from q-critic to q-target-critic
                for target_param, source_param in zip(self.networks[target_critic_key].parameters(),
                                                      self.networks[critic_key].parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

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

        return dict(q_critics=state_dict_critics)

    @override(TorchModel)
    def load_state_dict(self, state_dict: Dict) -> None:
        """implementation of :class:`~maze.core.agent.torch_model.TorchModel`
        """
        if "q_critics" in state_dict:
            state_dict_critics = state_dict["q_critics"]

            for key, critic in self.networks.items():
                assert key in state_dict_critics, f"Could not find state dict for critic ID: {key}"
                critic.load_state_dict(state_dict_critics[key])

    def re_init_networks(self) -> None:
        """Reinitialize all parameters of the network."""
        for key, critic in self.networks.items():
            # initialize model weights
            if isinstance(critic, InferenceBlock):
                critic.apply(make_module_init_normc(1.0))
                for block_key in critic.perception_dict:
                    if block_key == 'q_value' or block_key.endswith('_q_values'):
                        critic.perception_dict[block_key].apply(make_module_init_normc(0.01))
            else:
                inference_blocks = list(filter(lambda cc: isinstance(cc, InferenceBlock), critic.children()))
                if len(inference_blocks) == 1:
                    inference_blocks[0].apply(make_module_init_normc(1.0))
                    for block_key in inference_blocks[0].perception_dict:
                        if block_key == 'q_value' or block_key.endswith('_q_values'):
                            inference_blocks[0].perception_dict[block_key].apply(make_module_init_normc(0.01))
                else:
                    BColors.print_colored(f'More or less than one inference block was found for'
                                          f' {key}, therefore the model could not be reinitialized', BColors.WARNING)


class TorchSharedStateActionCritic(TorchStateActionCritic):
    """One critic is shared across all sub-steps or actors (default to use for standard gym-style environments).
    Can be instantiated via the
    :class:`~maze.perception.models.critics.shared_state_action_critics_composer.SharedStateActionCriticComposer`.
    """

    @override(TorchStateActionCritic)
    def predict_q_values(self, observations: Dict[Union[str, int], Dict[str, torch.Tensor]],
                         actions: Dict[Union[str, int], Dict[str, torch.Tensor]], gather_output: bool) -> \
            Dict[Union[str, int], List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        """implementation of
        :class:`~maze.core.agent.torch_state_action_critic.TorchStateActionCritic`
        """

        flattened_observations = flatten_spaces(observations.values())
        flattened_actions = flatten_spaces(actions.values())

        assert len(self.step_critic_keys) == 1
        step_id = self.step_critic_keys[0]
        if all(self.only_discrete_spaces.values()):
            out = self.compute_state_action_values_step(flattened_observations, step_id)
            # output shape List[Dict[str, (rollout_length, batch_dim)]]
            if gather_output:
                out = [{action_key: action_value.gather(-1, flattened_actions[
                    action_key.replace('_q_values', '')].long().unsqueeze(-1)).squeeze(-1)
                        for action_key, action_value in critic_out.items()} for critic_out in out]
            q_value = out
        else:
            q_value = self.compute_state_action_value_step(flattened_observations, flattened_actions, step_id)

        q_values = {step_id: q_value}
        return q_values

    @override(TorchStateActionCritic)
    def predict_next_q_values(self, next_observations: Dict[Union[str, int], Dict[str, torch.Tensor]],
                              next_actions: Dict[Union[str, int], Dict[str, torch.Tensor]],
                              next_actions_logits: Dict[Union[str, int], Dict[str, torch.Tensor]],
                              next_actions_log_probs: Dict[Union[str, int], Dict[str, torch.Tensor]],
                              alpha: Dict[Union[str, int], torch.Tensor]) \
            -> Dict[Union[str, int], Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """implementation of
        :class:`~maze.core.agent.torch_state_action_critic.TorchStateActionCritic`
        """

        flattened_next_observations = flatten_spaces(next_observations.values())
        flattened_next_actions = flatten_spaces(next_actions.values())
        flattened_next_actions_logits = flatten_spaces(next_actions_logits.values())
        flattened_next_action_log_probs = flatten_spaces(next_actions_log_probs.values())

        assert len(self.step_critic_keys) == 1
        step_id = self.step_critic_keys[0]
        alpha = sum(alpha.values())

        if all(self.only_discrete_spaces.values()):
            next_q_values = self.compute_state_action_values_step(flattened_next_observations,
                                                                  critic_id=(step_id, self.target_key))
            transpose_next_q_value = {k: [dic[k] for dic in next_q_values] for k in next_q_values[0]}
            next_q_value = dict()
            for q_action_head, q_values in transpose_next_q_value.items():
                action_key = q_action_head.replace('_q_values', '')
                tmp_q_value = torch.stack(q_values).min(dim=0).values
                next_action_probs = logits_to_probs(flattened_next_actions_logits[action_key])
                next_action_log_probs = torch.log(next_action_probs + (next_action_probs == 0.0).float() * 1e-8)

                # output shape of V(st) is (rollout_length, batch_dim)
                next_q_value[action_key] = torch.matmul(
                    next_action_probs.unsqueeze(-2),
                    (tmp_q_value - alpha * next_action_log_probs).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        else:
            next_q_value = self.compute_state_action_value_step(flattened_next_observations,
                                                                flattened_next_actions,
                                                                (step_id, self.target_key))
            next_q_value = torch.stack(next_q_value).min(dim=0).values - alpha * \
                            torch.stack(list(flattened_next_action_log_probs.values())).mean(dim=0)

        return {step_id: next_q_value}

    @property
    @override(TorchStateActionCritic)
    def num_critics(self) -> int:
        """implementation of
        :class:`~maze.core.agent.torch_state_action_critic.TorchStateActionCritic`
        """
        return 1


class TorchStepStateActionCritic(TorchStateActionCritic):
    """Each sub-step or actor gets its individual critic.
    Can be instantiated via the
    :class:`~maze.perception.models.critics.step_state_action_critic_composer.StepStateActionCriticComposer`.
    """

    @override(TorchStateActionCritic)
    def predict_q_values(self, observations: Dict[Union[str, int], Dict[str, torch.Tensor]],
                         actions: Dict[Union[str, int], Dict[str, torch.Tensor]], gather_output: bool) -> \
            Dict[Union[str, int], List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        """implementation of
        :class:`~maze.core.agent.torch_state_action_critic.TorchStateActionCritic`
        """

        q_values = dict()
        for step_id in observations.keys():
            if self.only_discrete_spaces[step_id]:
                out = self.compute_state_action_values_step(observations[step_id], step_id)
                # output shape List[Dict[str, (rollout_length, batch_dim)]]
                if gather_output:
                    out = [{action_key: action_value.gather(-1, actions[step_id][
                        action_key.replace('_q_values', '')].long().unsqueeze(-1)).squeeze(-1)
                            for action_key, action_value in critic_out.items()} for critic_out in out]
                q_values[step_id] = out
            else:
                q_values[step_id] = self.compute_state_action_value_step(observations[step_id], actions[step_id],
                                                                         step_id)

        return q_values

    @override(TorchStateActionCritic)
    def predict_next_q_values(self, next_observations: Dict[Union[str, int], Dict[str, torch.Tensor]],
                              next_actions: Dict[Union[str, int], Dict[str, torch.Tensor]],
                              next_actions_logits: Dict[Union[str, int], Dict[str, torch.Tensor]],
                              next_actions_log_probs: Dict[Union[str, int], Dict[str, torch.Tensor]],
                              alpha: Dict[Union[str, int], torch.Tensor]) -> Dict[
        Union[str, int], Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """implementation of
        :class:`~maze.core.agent.torch_state_action_critic.TorchStateActionCritic`
        """

        next_q_values = dict()
        for step_id in next_observations.keys():
            if self.only_discrete_spaces[step_id]:
                next_q_value = self.compute_state_action_values_step(next_observations[step_id],
                                                                     critic_id=(step_id, self.target_key))
                transpose_next_q_value = {k: [dic[k] for dic in next_q_value] for k in next_q_value[0]}
                next_q_values[step_id] = dict()
                for q_action_head, q_values in transpose_next_q_value.items():
                    action_key = q_action_head.replace('_q_values', '')
                    tmp_q_value = torch.stack(q_values).min(dim=0).values
                    next_action_probs = logits_to_probs(next_actions_logits[step_id][action_key])
                    next_action_log_probs = torch.log(next_action_probs + (next_action_probs == 0.0).float() * 1e-8)
                    # output shape of V(st) is (rollout_length, batch_dim)

                    next_q_values[step_id][action_key] = torch.matmul(
                        next_action_probs.unsqueeze(-2),
                        (tmp_q_value - alpha[step_id] * next_action_log_probs).unsqueeze(-1)).squeeze(-1).squeeze(-1)
            else:
                next_q_value = self.compute_state_action_value_step(next_observations[step_id],
                                                                    next_actions[step_id],
                                                                    (step_id, self.target_key))
                # output shape of V(st) is (rollout_length, batch_size)
                next_q_values[step_id] = torch.stack(next_q_value).min(dim=0).values - alpha[step_id] * \
                                         torch.stack(list(next_actions_log_probs[step_id].values())).mean(dim=0)

        return next_q_values

    @property
    @override(TorchStateActionCritic)
    def num_critics(self) -> int:
        """implementation of
        :class:`~maze.core.agent.torch_state_action_critic.TorchStateActionCritic`
        """
        return self.num_policies
