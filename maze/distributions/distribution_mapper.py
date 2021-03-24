"""Contains helper functions for working with ProbabilityDistributions"""
from typing import Dict, Tuple, Any, Sequence

import torch
from gym import spaces

from maze.core.utils.factory import Factory, CollectionOfConfigType
from maze.distributions.bernoulli import BernoulliProbabilityDistribution
from maze.distributions.categorical import CategoricalProbabilityDistribution
from maze.distributions.dict import DictProbabilityDistribution
from maze.distributions.gaussian import DiagonalGaussianProbabilityDistribution
from maze.distributions.multi_categorical import MultiCategoricalProbabilityDistribution
from maze.distributions.torch_dist import TorchProbabilityDistribution


class DistributionMapper:
    """Provides a mapping of spaces and action heads to the respective probability distributions to be used.

    This ensures full flexibility for specifying different distributions to the same gym action space type.
    (e.g. One gym.spaces.Box space could be modeled with a Beta another one with a DiagonalGaussian distribution.)
    It allows to add and register arbitrary custom distributions.

    :param action_space: The dictionary action space.
    :param distribution_mapper_config: A Distribution mapper configuration (for details see the docs).
    """

    # default space to distribution mapping
    default_mapping: Dict[type(spaces.Space), type(TorchProbabilityDistribution)] = dict()
    default_mapping[spaces.Discrete] = CategoricalProbabilityDistribution
    default_mapping[spaces.MultiBinary] = BernoulliProbabilityDistribution
    default_mapping[spaces.Box] = DiagonalGaussianProbabilityDistribution
    default_mapping[spaces.MultiDiscrete] = MultiCategoricalProbabilityDistribution

    def __init__(self, action_space: spaces.Dict, distribution_mapper_config: CollectionOfConfigType):
        self.action_space = action_space

        # mapping of action heads to distributions and configs
        self._action_head_to_distribution: Dict[str, Tuple[type(TorchProbabilityDistribution), Dict[str, Any]]] = dict()

        # first: apply default config to action heads
        for action_head, sub_action_space in action_space.spaces.items():
            space_type = type(sub_action_space)
            dist_type: TorchProbabilityDistribution = self.default_mapping[space_type]
            self._action_head_to_distribution[action_head] = (dist_type, {})

        # second: parse custom mappings
        for entry_dict in distribution_mapper_config:
            assert "distribution" in entry_dict
            assert ("action_space" in entry_dict and "action_head" not in entry_dict) or \
                   ("action_space" not in entry_dict and "action_head" in entry_dict)

            # get the distribution type
            distribution_type = Factory(TorchProbabilityDistribution).type_from_name(
                entry_dict["distribution"])

            # get additional distribution arguments
            args = entry_dict["args"] if "args" in entry_dict else {}

            if "action_head" in entry_dict:
                self._action_head_to_distribution[entry_dict["action_head"]] = (distribution_type, args)

            elif "action_space" in entry_dict:
                sub_action_space = Factory(spaces.Space).type_from_name(entry_dict["action_space"])

                for action_head in self.action_space.spaces:

                    if isinstance(self.action_space[action_head], sub_action_space):
                        self._action_head_to_distribution[action_head] = (distribution_type, args)

    def required_logits_shape(self, action_head: str) -> Sequence[int]:
        """Returns the required logits shape (network output shape) for a given action head.

        :param action_head: The name of the action head (action dictionary key).
        :return: The required logits shape.
        """
        # get distribution class and request required logits shape
        distribution_cls, _ = self._action_head_to_distribution[action_head]
        return distribution_cls.required_logits_shape(action_space=self.action_space[action_head])

    def action_head_distribution(self, action_head: str, logits: torch.Tensor,
                                 temperature: float) -> TorchProbabilityDistribution:
        """Creates a probability distribution for a given action head.

        :param action_head: The name of the action head (action dictionary key).
        :param logits: the logits to parameterize the distribution from
        :param temperature: Controls the sampling behaviour
                                * 1.0 corresponds to unmodified sampling
                                * smaller than 1.0 concentrates the action distribution towards deterministic sampling
        :return: (ProbabilityDistribution) the appropriate instance of a ProbabilityDistribution
        """

        # get distribution class and respective action space
        distribution_cls, args = self._action_head_to_distribution[action_head]
        action_space = self.action_space[action_head]

        # instantiate distribution instance
        return distribution_cls(logits=logits, action_space=action_space, temperature=temperature, **args)

    def logits_dict_to_distribution(self, logits_dict: Dict[str, torch.Tensor], temperature: float) \
            -> DictProbabilityDistribution:
        """Creates a dictionary probability distribution for a given logits dictionary.

        :param logits_dict: A logits dictionary [action_head: action_logits] to parameterize the distribution from.
        :param temperature: Controls the sampling behaviour.
                                * 1.0 corresponds to unmodified sampling
                                * smaller than 1.0 concentrates the action distribution towards deterministic sampling
        :return: (DictProbabilityDistribution) the respective instance of a DictProbabilityDistribution.
        """

        # iterate all action heads contained in logits dictionary
        distribution_dict = dict()
        for action_head, action_logits in logits_dict.items():
            assert isinstance(action_logits, torch.Tensor)
            distribution_dict[action_head] = self.action_head_distribution(action_head=action_head,
                                                                           logits=action_logits,
                                                                           temperature=temperature)

        return DictProbabilityDistribution(distribution_dict=distribution_dict)

    def __repr__(self):
        """Give a string representation of the distribution helper

        :return: A string representation
        """
        txt = '(Distribution mapper):'
        max_length = max(map(len, self._action_head_to_distribution.keys()))
        for action_head, (dist_type, args) in self._action_head_to_distribution.items():
            default_dist = self.default_mapping[type(self.action_space[action_head])]
            txt += f'\n\t{action_head}'.ljust(max_length + 2) + ' -> ' \
                                                                f'space: {self.action_space[action_head]}, ' \
                                                                f'used-dist: {dist_type.__name__}, ' \
                                                                f'shape: {self.required_logits_shape(action_head)}, ' \
                                                                f'args: {args}'
            if default_dist != dist_type:
                txt += f', [default-dist: {default_dist.__name__}]'
        return txt
