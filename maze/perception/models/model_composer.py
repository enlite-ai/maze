"""Interface for model composers,
encapsulating the set of policy and critic networks along with the distribution mapper."""
import os
from abc import abstractmethod, ABC
from typing import Dict, Optional

import gym

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_critic import TorchStateCritic
from maze.core.env.structured_env import StepKeyType
from maze.core.utils.factory import ConfigType
from maze.core.utils.structured_env_utils import flat_structured_space
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.blocks.inference import InferenceGraph, InferenceBlock
from maze.utils.bcolors import BColors


class BaseModelComposer(ABC):
    """Abstract baseclass and interface definitions for model composers.

    Model composers encapsulate the set of policy and critic networks along with the distribution mapper.

    :param action_spaces_dict: Dict of sub-step id to action space.
    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param distribution_mapper_config: Distribution mapper configuration.
    """

    @classmethod
    @abstractmethod
    def check_model_config(cls, model_config: ConfigType) -> None:
        """Asserts the provided model config for consistency.
        :param model_config: The model config to check.
        """

    def __init__(self,
                 action_spaces_dict: Dict[StepKeyType, gym.spaces.Dict],
                 observation_spaces_dict: Dict[StepKeyType, gym.spaces.Dict],
                 agent_counts_dict: Dict[StepKeyType, int],
                 distribution_mapper_config: ConfigType):
        self.action_spaces_dict = action_spaces_dict
        self.observation_spaces_dict = observation_spaces_dict
        self.agent_counts_dict = agent_counts_dict

        # initialize DistributionMapper
        flat_action_space = flat_structured_space(action_spaces_dict)
        self._distribution_mapper = DistributionMapper(action_space=flat_action_space,
                                                       distribution_mapper_config=distribution_mapper_config)
        self._network_image_paths = None

    @property
    @abstractmethod
    def policy(self) -> Optional[TorchPolicy]:
        """Policy networks."""

    @property
    @abstractmethod
    def critic(self) -> Optional[TorchStateCritic]:
        """The critic model."""

    @property
    def distribution_mapper(self) -> DistributionMapper:
        """The DistributionMapper, mapping the action heads to distributions."""
        return self._distribution_mapper

    @property
    def network_image_paths(self) -> Optional[Dict[str, Dict[str, str]]]:
        """The paths of the individual pickled network images figures."""
        return self._network_image_paths

    def save_models(self) -> None:
        """Save the policies and critics as pdfs."""

        def plot_inference_graphs(nets_type, nets) -> Dict[str, str]:
            """Draw inference graphs."""
            saved_models = dict()
            for net_name, net_model in nets.items():
                if not os.path.exists(f'{nets_type}_{net_name}'):
                    if isinstance(net_model, InferenceBlock):
                        path = InferenceGraph(net_model).save(f'{nets_type}_{net_name}', './', save_pickle_figure=True)
                        saved_models[net_name] = path
                    else:
                        children = net_model.children()
                        inference_blocks = list(filter(lambda cc: isinstance(cc, InferenceBlock), children))
                        if len(inference_blocks) == 1:
                            path = InferenceGraph(inference_blocks[0]).save(f'{nets_type}_{net_name}', './', True)
                            saved_models[net_name] = path
                        elif len(inference_blocks) > 1:
                            BColors.print_colored(f'More than one inference block was found for'
                                                  f' {nets_type}-{net_name}, please revisit the model and make '
                                                  f'sure only one is present', BColors.WARNING)
                        else:
                            BColors.print_colored(f'No inference block could be found in '
                                                  f'{nets_type}-{net_name}, thus no visual representation '
                                                  f'(of the model) could be created or saved', BColors.WARNING)
            return saved_models

        self._network_image_paths = dict()
        try:
            if self.policy:
                self._network_image_paths['policy'] = plot_inference_graphs("policy", self.policy.networks)
            if self.critic:
                self._network_image_paths['critic'] = plot_inference_graphs("critic", self.critic.networks)
        except ImportError as e:
            BColors.print_colored(f'Models graphical representation could not be saved: {e}',
                                  BColors.WARNING)
