""" Template network building functions. """
import functools
from typing import Optional, Dict, Union, Type, Tuple, List

from gym import spaces

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_action_critic import TorchSharedStateActionCritic, \
    TorchStateActionCritic, TorchStepStateActionCritic
from maze.core.agent.torch_state_critic import TorchStateCritic, \
    TorchSharedStateCritic, TorchStepStateCritic, TorchDeltaStateCritic
from maze.core.annotations import override
from maze.core.env.structured_env import StepKeyType
from maze.core.utils.factory import ConfigType, Factory
from maze.core.utils.structured_env_utils import flat_structured_space
from maze.core.wrappers.observation_preprocessing.preprocessors.one_hot import OneHotPreProcessor
from maze.perception.blocks.inference import InferenceBlock
from maze.perception.blocks.output.linear import LinearOutputBlock
from maze.perception.builders.base import BaseModelBuilder
from maze.perception.models.critics import SharedStateCriticComposer, \
    DeltaStateCriticComposer, StepStateCriticComposer
from maze.perception.models.critics.critic_composer_interface import CriticComposerInterface
from maze.perception.models.critics.shared_state_action_critics_composer import SharedStateActionCriticComposer
from maze.perception.models.critics.step_state_action_critic_composer import StepStateActionCriticComposer
from maze.perception.models.model_composer import BaseModelComposer
from maze.perception.models.policies.base_policy_composer import BasePolicyComposer
from maze.perception.models.policies.probabilistic_policy_composer import ProbabilisticPolicyComposer
from maze.perception.weight_init import make_module_init_normc
from maze.utils.bcolors import BColors


class TemplateModelComposer(BaseModelComposer):
    """Composes template models from configs.

    :param action_spaces_dict: Dict of sub-step id to action space.
    :param observation_spaces_dict: Dict of sub-step id to observation space.
    :param distribution_mapper_config: Distribution mapper configuration.
    :param model_builder: The model builder (template) to create the model from.
    :param policy: Specifies the policy type as a configType.
        E.g. {'type': maze.perception.models.policies.ProbabilisticPolicyComposer} specifies a probabilistic policy.
    :param critic: Specifies the critic type as a configType.
        E.g. {'type': maze.perception.models.critics.StateCriticComposer} specifies the single step state critic.
    """

    @classmethod
    def check_model_config(cls, model_config: ConfigType) -> None:
        """Asserts the provided model config for consistency.
        :param model_config: The model config to check.
        """
        if 'policy' in model_config:
            assert '_target_' in model_config['policy']
            assert 'networks' not in model_config['policy'], \
                f"Template models do not expect explicit policy networks! Check the model config!"
        if 'critic' in model_config and model_config["critic"]:
            assert '_target_' in model_config['critic']
            assert 'networks' not in model_config['critic'], \
                f"Template models do not expect explicit critic networks! Check the model config!"

    def __init__(self,
                 action_spaces_dict: Dict[StepKeyType, spaces.Dict],
                 observation_spaces_dict: Dict[StepKeyType, spaces.Dict],
                 agent_counts_dict: Dict[StepKeyType, int],
                 distribution_mapper_config: ConfigType,
                 model_builder: Union[ConfigType, Type[BaseModelBuilder]],
                 policy: ConfigType,
                 critic: ConfigType):

        super().__init__(action_spaces_dict, observation_spaces_dict, agent_counts_dict, distribution_mapper_config)

        self._policy_type = Factory(BasePolicyComposer).type_from_name(policy['_target_']) \
            if policy is not None else None
        self._critic_type = Factory(CriticComposerInterface).type_from_name(critic['_target_']) \
            if critic is not None else None
        self.model_builder = Factory(BaseModelBuilder).instantiate(model_builder)

        self.model_builder.init_shared_embedding_keys(self.action_spaces_dict.keys())
        assert self.model_builder.shared_embedding_keys.keys() == self.observation_spaces_dict.keys()
        self._shared_embedding_nets: Dict[StepKeyType, InferenceBlock] = dict()

        self.save_models()

    def template_perception_net(self, observation_space: spaces.Dict) -> InferenceBlock:
        """Compiles a template perception network for a given observation space.

        :param observation_space: The observation space tp build the model for.
        :return: A Perception Inference Block.
        """

        # build model from parameters
        perception_net = self.model_builder.from_observation_space(observation_space=observation_space)

        # initialize model weights
        module_init = make_module_init_normc(std=1.0)
        perception_net.apply(module_init)

        return perception_net

    def template_policy_net(self, observation_space: spaces.Dict, action_space: spaces.Dict,
                            shared_embedding_keys: List[str]) -> Tuple[InferenceBlock, InferenceBlock]:
        """Compiles a template policy network.

        :param observation_space: The input observations for the perception network.
        :param action_space: The action space that defines the network action heads.
        :param shared_embedding_keys: The list of embedding keys for this substep's model.
        :return: A policy network (actor) InferenceBlock, as well as the embedding net InferenceBlock if shared keys
            have been specified.
        """

        # build perception net
        embedding_net = self.template_perception_net(observation_space)

        # build action head
        perception_dict = embedding_net.perception_dict
        action_heads = []
        for action_head, action_space in action_space.spaces.items():
            # initialize action head
            action_net = LinearOutputBlock(in_keys="latent", out_keys=action_head,
                                           in_shapes=perception_dict["latent"].out_shapes(),
                                           output_units=self._distribution_mapper.required_logits_shape(action_head)[0])

            module_init = make_module_init_normc(std=0.01)
            action_net.apply(module_init)

            # extent perception dictionary
            perception_dict[action_head] = action_net
            action_heads.append(action_head)

        # compile inference model
        shared_embedding_keys_remove_input = list(filter(lambda x: x not in embedding_net.in_keys,
                                                         shared_embedding_keys))
        net = InferenceBlock(in_keys=embedding_net.in_keys, out_keys=action_heads + shared_embedding_keys_remove_input,
                             in_shapes=embedding_net.in_shapes, perception_blocks=perception_dict)

        if len(shared_embedding_keys_remove_input) == 0:
            embedding_net = None

        return net, embedding_net

    def template_value_net(self,
                           observation_space: Optional[spaces.Dict],
                           shared_embedding_keys: List[str] = None,
                           perception_net: Optional[InferenceBlock] = None) -> InferenceBlock:
        """Compiles a template value network.

        :param observation_space: The input observations for the perception network.
        :param shared_embedding_keys: The shared embedding keys for this substep's model (input)
        :param perception_net: The embedding network of the policy network if shared keys have been specified, in order
            reuse the block and share the embedding.
        :return: A value network (critic) InferenceBlock.
        """

        shared_embedding_keys = [] if shared_embedding_keys is None else shared_embedding_keys
        # # build perception net
        if perception_net is None:
            assert len(shared_embedding_keys) == 0
            perception_net = self.template_perception_net(observation_space)
            perception_dict = perception_net.perception_dict
            inference_block_in_keys = perception_net.in_keys
            inference_block_in_shapes = perception_net.in_shapes
        else:
            assert len(shared_embedding_keys) > 0
            perception_dict = dict()
            _block_not_present_in_perception_net = list(filter(lambda x: x not in perception_net.perception_dict,
                                                               shared_embedding_keys))
            assert len(_block_not_present_in_perception_net) == 0, \
                f'All given latent keys, must be present in the default perception dict (created for the policy).' \
                f'But the following keys {_block_not_present_in_perception_net} was not found.'
            _blocks_with_more_outputs = list(filter(lambda x: len(perception_net.perception_dict[x].out_shapes()) > 1,
                                                    shared_embedding_keys))
            assert len(_blocks_with_more_outputs) == 0, \
                f'All given latent keys, must refer to a block with only a single output. But the following keys ' \
                f'have more than one output {_blocks_with_more_outputs}.'

            new_perception_net = self.template_perception_net(observation_space)
            new_perception_dict = new_perception_net.perception_dict
            start_adding_blocks = False
            for idx, block_keys in perception_net.execution_plan.items():
                if start_adding_blocks:
                    for block_key in block_keys:
                        perception_dict[block_key] = new_perception_dict[block_key]
                if any([block_key in shared_embedding_keys for block_key in block_keys]):
                    assert all([[block_key in shared_embedding_keys for block_key in block_keys]]), \
                        f'In the template model composer, all shared embedding keys given must be on the same level' \
                        f'in the execution plan of the inference block.'
                    start_adding_blocks = True

            inference_block_in_keys = shared_embedding_keys
            inference_block_in_shapes = sum([perception_net.perception_dict[key].out_shapes() for key in
                                             shared_embedding_keys], [])

        # build value head
        value_net = LinearOutputBlock(in_keys="latent", out_keys="value",
                                      in_shapes=perception_net.perception_dict["latent"].out_shapes(),
                                      output_units=1)

        module_init = make_module_init_normc(std=0.01)
        value_net.apply(module_init)

        # extent perception dictionary
        perception_dict["value"] = value_net

        # compile inference model
        net = InferenceBlock(in_keys=inference_block_in_keys, out_keys="value", in_shapes=inference_block_in_shapes,
                             perception_blocks=perception_dict)

        return net

    def template_q_value_net(self,
                             observation_space: Optional[spaces.Dict],
                             action_space: spaces.Dict,
                             only_discrete_spaces: bool,
                             perception_net: Optional[InferenceBlock] = None) -> InferenceBlock:
        """Compiles a template state action (Q) value network.

        :param observation_space: The input observations for the perception network.
        :param action_space: The action space that defines the network action heads.
        :param perception_net: A initial network to continue from.
                               (e.g. useful for shared weights. Model building continues from the key 'latent'.)
        :param only_discrete_spaces: A dict specifying if the action spaces w.r.t. the step only hold discrete action
                                     spaces.
        :return: A q value network (critic) InferenceBlock.
        """
        assert all(map(lambda space: isinstance(space, (spaces.Discrete, spaces.Box)),
                       action_space.spaces.values())), 'Only discrete and box spaces supported thus far for q values ' \
                                                       'critic.'

        if not only_discrete_spaces:
            discrete_space = list(filter(lambda kk: isinstance(action_space.spaces[kk], spaces.Discrete),
                                         action_space.spaces))
            if len(discrete_space) > 0:
                new_action_space = {}
                for key in action_space.spaces.keys():
                    if key in discrete_space:
                        new_action_space[key] = OneHotPreProcessor(action_space.spaces[key]).processed_space()
                    else:
                        new_action_space[key] = action_space.spaces[key]
                action_space = spaces.Dict(new_action_space)
            observation_space = spaces.Dict({**observation_space.spaces, **action_space.spaces})
            value_heads = {'q_value': 1}
        else:
            value_heads = {f'{act_key}_q_values': act_space.n for act_key, act_space in action_space.spaces.items()}

        # check if actions are considered as observations for the state-action critic
        for action_head in action_space.spaces.keys():
            if action_head not in self.model_builder.observation_modality_mapping:
                BColors.print_colored(
                    f'TemplateModelComposer: The action \'{action_head}\' could not be found in the '
                    f'model_builder.observation_modality_mapping and wont be considered '
                    f'as an input to the state-action critic!',
                    BColors.FAIL)

        # build perception net
        if perception_net is None:
            perception_net = self.template_perception_net(observation_space)

        perception_dict = perception_net.perception_dict
        for value_head, output_units in value_heads.items():
            # initialize action head
            value_net = LinearOutputBlock(in_keys="latent", out_keys=value_head,
                                          in_shapes=perception_dict["latent"].out_shapes(),
                                          output_units=output_units)

            module_init = make_module_init_normc(std=0.01)
            value_net.apply(module_init)

            # extent perception dictionary
            perception_dict[value_head] = value_net

        # compile inference model
        net = InferenceBlock(in_keys=perception_net.in_keys, out_keys=list(value_heads.keys()),
                             in_shapes=perception_net.in_shapes, perception_blocks=perception_dict)

        return net

    def _only_discrete_spaces(self) -> Dict[Union[str, int], bool]:
        """Check if the actions spaces have only discrete spaces.

        :return: A dict holding a bool indicating whether only discrete spaces are present w.r.t. to the steps.
        """
        only_discrete_spaces = {step_key: True for step_key in self.action_spaces_dict.keys()}
        for step_key, dict_action_space in self.action_spaces_dict.items():
            for act_key, act_space in dict_action_space.spaces.items():
                if only_discrete_spaces[step_key] and not isinstance(act_space, spaces.Discrete):
                    only_discrete_spaces[step_key] = False
        return only_discrete_spaces

    @property
    @functools.lru_cache()
    @override(BaseModelComposer)
    def policy(self) -> Optional[TorchPolicy]:
        """Implementation of the BaseModelComposer interface, returns the policy networks."""

        if self._policy_type is None:
            return None

        elif issubclass(self._policy_type, ProbabilisticPolicyComposer):
            networks = dict()
            for sub_step_key in self.action_spaces_dict.keys():
                networks[sub_step_key], self._shared_embedding_nets[sub_step_key] = self.template_policy_net(
                    observation_space=self.observation_spaces_dict[sub_step_key],
                    action_space=self.action_spaces_dict[sub_step_key],
                    shared_embedding_keys=self.model_builder.shared_embedding_keys[sub_step_key])

            return TorchPolicy(networks=networks, distribution_mapper=self.distribution_mapper, device="cpu")

        else:
            raise ValueError(f"Policy type {self._policy_type} not supported by the template model composer!")

    @property
    @functools.lru_cache()
    @override(BaseModelComposer)
    def critic(self) -> Optional[Union[TorchStateCritic, TorchStateActionCritic]]:
        """Implementation of the BaseModelComposer interface, returns the value networks."""

        if self._critic_type is None:
            return None

        elif issubclass(self._critic_type, SharedStateCriticComposer):
            assert not any(self.model_builder.use_shared_embedding.values()), \
                f'Embedding can not be shared when using a shared state critic'
            observation_space = flat_structured_space(self.observation_spaces_dict)
            critics = {0: self.template_value_net(observation_space)}
            return TorchSharedStateCritic(networks=critics, obs_spaces_dict=self.observation_spaces_dict, device="cpu",
                                          stack_observations=False)

        elif issubclass(self._critic_type, StepStateCriticComposer):
            critics = dict()
            for sub_step_key, sub_step_space in self.observation_spaces_dict.items():
                critics[sub_step_key] = self.template_value_net(
                    observation_space=sub_step_space, perception_net=self._shared_embedding_nets[sub_step_key],
                    shared_embedding_keys=self.model_builder.shared_embedding_keys[sub_step_key])
            return TorchStepStateCritic(networks=critics,
                                        obs_spaces_dict=self.observation_spaces_dict,
                                        device="cpu")

        elif issubclass(self._critic_type, DeltaStateCriticComposer):
            critics = dict()
            for idx, (sub_step_key, sub_step_space) in enumerate(self.observation_spaces_dict.items()):
                if idx > 0:
                    sub_step_space = spaces.Dict({**sub_step_space.spaces,
                                                  **DeltaStateCriticComposer.prev_value_space.spaces})
                critics[sub_step_key] = self.template_value_net(
                    observation_space=sub_step_space, perception_net=self._shared_embedding_nets[sub_step_key],
                    shared_embedding_keys=self.model_builder.shared_embedding_keys[sub_step_key])

            return TorchDeltaStateCritic(networks=critics,
                                         obs_spaces_dict=self.observation_spaces_dict,
                                         device="cpu")
        elif issubclass(self._critic_type, SharedStateActionCriticComposer):
            assert not any(self.model_builder.use_shared_embedding.values()), \
                f'Embedding can not be shared when using a shared state action critic'

            observation_space = flat_structured_space(self.observation_spaces_dict)
            action_space = flat_structured_space(self.action_spaces_dict)
            only_discrete_spaces = self._only_discrete_spaces()
            critics = {0: self.template_q_value_net(observation_space, action_space,
                                                    all(only_discrete_spaces.values()))}
            return TorchSharedStateActionCritic(networks=critics,
                                                num_policies=len(self.action_spaces_dict),
                                                device="cpu",
                                                only_discrete_spaces=only_discrete_spaces,
                                                action_spaces_dict={0: action_space})

        elif issubclass(self._critic_type, StepStateActionCriticComposer):
            assert not any(self.model_builder.use_shared_embedding.values()), \
                f'Embedding can not be shared when using a step state action critic'
            critics = dict()
            only_discrete_spaces = self._only_discrete_spaces()
            for sub_step_key, sub_step_space in self.observation_spaces_dict.items():
                critics[sub_step_key] = self.template_q_value_net(observation_space=sub_step_space,
                                                                  action_space=self.action_spaces_dict[sub_step_key],
                                                                  only_discrete_spaces=only_discrete_spaces[
                                                                      sub_step_key])
            return TorchStepStateActionCritic(networks=critics,
                                              num_policies=len(self.action_spaces_dict),
                                              device="cpu", only_discrete_spaces=only_discrete_spaces,
                                              action_spaces_dict=self.action_spaces_dict)

        else:
            raise ValueError(f"Critics type {self._critic_type} not supported by the template model composer!")
