# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch version of the functions to compute V-trace off-policy actor critic
targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.

In addition to the original paper's code, changes have been made
to support MultiDiscrete action spaces. behaviour_policy_logits,
target_policy_logits and actions parameters in the entry point
multi_from_logits method accepts lists of tensors instead of just
tensors.
"""
import collections
from typing import List, Tuple, Union

import torch

from maze.core.env.action_conversion import TorchActionType
from maze.distributions.dict import DictProbabilityDistribution
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.perception_utils import convert_to_torch

VTraceFromLogitsReturns = collections.namedtuple(
    'VTraceFromLogitsReturns',
    ['vs', 'pg_advantages', 'log_rhos',
     'behaviour_action_log_probs', 'target_action_log_probs', 'target_step_action_dists'])

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


def log_probs_from_logits_and_actions_and_spaces(
        policy_logits: List[TorchActionType],
        actions: List[TorchActionType],
        distribution_mapper: DistributionMapper) \
        -> Tuple[List[TorchActionType], List[DictProbabilityDistribution]]:
    """Computes action log-probs from policy logits, actions and acton_spaces.

    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and
    NUM_ACTIONS refers to the number of actions.

    :param policy_logits: A list (w.r.t. the substeps of the env) of dicts (w.r.t. the actions) of tensors
        of un-normalized log-probabilities (shape list[dict[str,[T, B, NUM_ACTIONS]]])
    :param actions: An list (w.r.t. the substeps of the env) of dicts (w.r.t. the actions) of tensors
        (list[dict[str,[T, B]]])
    :param distribution_mapper: A distribution mapper providing a mapping of action heads to distributions.

    :return: A list (w.r.t. the substeps of the env) of dicts (w.r.t. the actions) of tensors of shape [T, B]
        corresponding to the sampling log probability of the chosen action w.r.t. the policy.
        And a list (w.r.t. the substeps of the env) of DictProbability distributions corresponding to the step-action-
        distributions.
    """
    log_probs = list()
    step_action_dists = list()
    for step_policy_logits, step_actions in zip(policy_logits, actions):
        step_action_dist = distribution_mapper.logits_dict_to_distribution(logits_dict=step_policy_logits,
                                                                           temperature=1.0)
        log_probs.append(step_action_dist.log_prob(step_actions))
        step_action_dists.append(step_action_dist)
    return log_probs, step_action_dists


def from_logits(behaviour_policy_logits: List[TorchActionType],
                target_policy_logits: List[TorchActionType],
                actions: List[TorchActionType],
                distribution_mapper: DistributionMapper,
                discounts: torch.Tensor,
                rewards: torch.Tensor,
                values: List[torch.Tensor],
                bootstrap_value: List[torch.Tensor],
                clip_rho_threshold: Union[float, None],
                clip_pg_rho_threshold: Union[float, None],
                device: Union[str, None]) -> VTraceFromLogitsReturns:
    r"""V-trace for softmax policies.

    Calculates V-trace actor critic targets for softmax polices as described in

    "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al.

    Target policy refers to the policy we are interested in improving and
    behaviour policy refers to the policy that generated the given
    rewards and actions.

    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and
    ACTION_SPACE refers to the list of numbers each representing a number of
    actions.

    :param behaviour_policy_logits: A list (w.r.t. the substeps of the env) of dict (w.r.t. the actions) of tensors
                                    of un-normalized log-probabilities (shape list[dict[str,[T, B, NUM_ACTIONS]]])
    :param target_policy_logits: A list (w.r.t. the substeps of the env) of dict (w.r.t. the actions) of tensors
                                 of un-normalized log-probabilities (shape list[dict[str,[T, B, NUM_ACTIONS]]])
    :param actions: An list (w.r.t. the substeps of the env) of dicts (w.r.t. the actions) with actions sampled from
                    the behavior policy. (list[dict[str,[T, B]]])
    :param distribution_mapper: A distribution mapper providing a mapping of action heads to distributions.
    :param discounts: A float32 tensor of shape [T, B] with the discount encountered when following the behavior policy.
    :param rewards: A float32 tensor of shape [T, B] with the rewards generated by following the behavior policy.
    :param values: A list (w.r.t. the substeps of the env) of float32 tensors of shape [T, B] with the value function
                   estimates wrt. the target policy.
    :param bootstrap_value: A list (w.r.t. the substeps of the env) of float32 tensors of shape [B] with the value
                            function estimate at time T.
    :param clip_rho_threshold: A scalar float32 tensor with the clipping threshold for importance weights (rho) when
                               calculating the baseline targets (vs). rho^bar in the paper.
    :param clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold on rho_s in:
                                  \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)).
    :param device: the device the results should be sent to before returning it

    :return: A `VTraceFromLogitsReturns` namedtuple with the following fields:
             vs: A list (w.r.t. the substeps of the env) of float32 tensors of shape [T, B]. Can be used as target to
             train a baseline (V(x_t) - vs_t)^2.
             pg_advantages: A list (w.r.t. the substeps of the env) of float32 tensors of shape [T, B]. Can be used as
             an estimate of the advantage in the calculation of policy gradients.
             log_rhos: A list (w.r.t. the substeps of the env) of float32 tensors of shape [T, B] containing the log
             importance sampling weights (log rhos).
             behaviour_action_log_probs: A list (w.r.t. the substeps of the env) of float32 tensors of shape [T, B]
             containing the behaviour policy action log probabilities (log \mu(a_t)).
             target_action_log_probs: A list (w.r.t. the substeps of the env) of float32 tensors of shape [T, B]
             containing target policy action probabilities (log \pi(a_t)).
             target_step_action_dists: A list (w.r.t. the substeps of the env) of the action probability distributions
             w.r.t. to the target policy
    """

    behaviour_action_log_probs, _ = \
        log_probs_from_logits_and_actions_and_spaces(behaviour_policy_logits, actions, distribution_mapper)
    target_action_log_probs, target_step_action_dists = \
        log_probs_from_logits_and_actions_and_spaces(target_policy_logits, actions, distribution_mapper)

    log_rhos = get_log_rhos(target_action_log_probs=target_action_log_probs,
                            behaviour_action_log_probs=behaviour_action_log_probs)
    vss = list()
    pg_advantagess = list()
    for step_log_rhos, step_values, step_bootstrap_values in zip(log_rhos, values, bootstrap_value):
        vs, pg_advantages = from_importance_weights(
            log_rhos=step_log_rhos,
            discounts=discounts,
            rewards=rewards,
            values=step_values,
            bootstrap_value=step_bootstrap_values,
            clip_rho_threshold=clip_rho_threshold,
            clip_pg_rho_threshold=clip_pg_rho_threshold)
        vss.append(vs.to(device) if device is not None else vs)
        pg_advantagess.append(pg_advantages.to(device) if device is not None else pg_advantages)

    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behaviour_action_log_probs=behaviour_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        vs=vss, pg_advantages=pg_advantagess, target_step_action_dists=target_step_action_dists)


def from_importance_weights(log_rhos: torch.Tensor,
                            discounts: torch.Tensor,
                            rewards: torch.Tensor,
                            values: torch.Tensor,
                            bootstrap_value: torch.Tensor,
                            clip_rho_threshold: Union[float, None],
                            clip_pg_rho_threshold: Union[float, None]) -> VTraceReturns:
    r"""V-trace from log importance weights.

    Calculates V-trace actor critic targets as described in

    "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al.

    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size. This code
    also supports the case where all tensors have the same number of additional
    dimensions, e.g., `rewards` is [T, B, C], `values` is [T, B, C],
    `bootstrap_value` is [B, C].

    :param log_rhos: A float32 tensor of shape [T, B] representing the log
                     importance sampling weights, i.e. log(target_policy(a) / behaviour_policy(a)).
                     V-trace performs operations on rhos in log-space for numerical stability.
    :param discounts: A float32 tensor of shape [T, B] with discounts encountered when following the behaviour policy.
    :param rewards: A float32 tensor of shape [T, B] containing rewards generated by following the behaviour policy.
    :param values: A float32 tensor of shape [T, B] with the value function estimates wrt. the target policy.
    :param bootstrap_value: A float32 of shape [B] with the value function estimate at time T.
    :param clip_rho_threshold: A scalar float32 tensor with the clipping threshold for importance weights (rho) when
                               calculating the baseline targets (vs). rho^bar in the paper. If None, no clipping is
                               applied.
    :param clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold on rho_s in
                                  \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_sfrom_importance_weights)).
                                  If None, no clipping is applied.

    :return: A VTraceReturns namedtuple (vs, pg_advantages) where:
             vs: A float32 tensor of shape [T, B]. Can be used as target to train a baseline (V(x_t) - vs_t)^2.
             pg_advantages: A float32 tensor of shape [T, B]. Can be used as the advantage in the calculation of policy
             gradients.
    """
    log_rhos = convert_to_torch(log_rhos, device='cpu', cast=torch.float32, in_place='try')
    discounts = convert_to_torch(discounts, device='cpu', cast=torch.float32, in_place='try')
    rewards = convert_to_torch(rewards, device='cpu', cast=torch.float32, in_place='try')
    values = convert_to_torch(values, device='cpu', cast=torch.float32, in_place='try')
    bootstrap_value = convert_to_torch(bootstrap_value, device='cpu', cast=torch.float32, in_place='try')

    # Make sure tensor ranks are consistent.
    rho_rank = len(log_rhos.size())  # Usually 2.
    assert rho_rank == len(values.size())
    assert rho_rank - 1 == len(bootstrap_value.size()), \
        "must have rank {}".format(rho_rank - 1)
    assert rho_rank == len(discounts.size())
    assert rho_rank == len(rewards.size())

    rhos = torch.exp(log_rhos)
    if clip_rho_threshold is not None:
        # Basically the min operation rho := min(rho_bar, pi(a_t|x_t)/mu(at|x_t))
        clipped_rhos = torch.clamp_max(rhos, clip_rho_threshold)
    else:
        clipped_rhos = rhos

    cs = torch.clamp_max(rhos, 1.0)
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat(
        [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    vs_minus_v_xs: List = [torch.zeros_like(bootstrap_value)]
    for i in reversed(range(len(discounts))):
        discount_t, c_t, delta_t = discounts[i], cs[i], deltas[i]
        vs_minus_v_xs.append(delta_t + discount_t * c_t * vs_minus_v_xs[-1])
    vs_minus_v_xs: torch.Tensor = torch.stack(vs_minus_v_xs[1:])
    # Reverse the results back to original order.
    vs_minus_v_xs: torch.Tensor = torch.flip(vs_minus_v_xs, dims=[0])
    # Add V(x_s) to get v_s.
    vs = vs_minus_v_xs + values

    # Advantage for policy gradient.
    vs_t_plus_1 = torch.cat(
        [vs[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.clamp_max(rhos, clip_pg_rho_threshold)
    else:
        clipped_pg_rhos = rhos
    pg_advantages = (
            clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))

    # Make sure no gradients backpropagated through the returned values.
    return VTraceReturns(vs=vs.detach(), pg_advantages=pg_advantages.detach())


def get_log_rhos(target_action_log_probs: List[TorchActionType],
                 behaviour_action_log_probs: List[TorchActionType]) \
        -> List[torch.Tensor]:
    """With the selected log_probs for multi-discrete actions of behavior
    and target policies we compute the log_rhos for calculating the vtrace.

    :param target_action_log_probs: A list (w.r.t. the substeps of the env) of dicts (w.r.t. the actions) of tensors of
        shape [T, B] corresponding to the sampling log probability of the chosen action w.r.t. the target policy.
    :param behaviour_action_log_probs: A list (w.r.t. the substeps of the env) of dicts (w.r.t. the actions) of tensors
        of shape [T, B] corresponding to the sampling log probability of the chosen action w.r.t. the behaviour policy.

    :return: a list (w.r.t. the substeps of the env) of tensors, where each tensor is of the shape [T,B]
    """
    log_rhos = list()
    # TODO: Consider doing this for each individual action
    with torch.no_grad():
        for step_target_action_log_probs, step_behaviour_action_log_probs in zip(target_action_log_probs,
                                                                                 behaviour_action_log_probs):
            target = torch.stack(list(step_target_action_log_probs.values()))
            behaviour = torch.stack(list(step_behaviour_action_log_probs.values()))

            step_log_rhos = torch.sum(target - behaviour, dim=0)
            log_rhos.append(step_log_rhos)
    return log_rhos
