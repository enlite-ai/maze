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

"""Tests for V-trace.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

from maze.distributions.distribution_mapper import DistributionMapper
from maze.train.trainers.impala import impala_vtrace
from maze.perception.perception_utils import convert_to_torch


def _shaped_arange(*shape):
    """Runs np.arange, converts to float and reshapes."""
    return np.arange(int(np.prod(shape)), dtype=np.float32).reshape(*shape)


def _softmax(logits):
    """Applies softmax non-linearity on inputs."""
    return np.exp(logits) / np.sum(np.exp(np.array(logits)), axis=-1, keepdims=True)


def _ground_truth_calculation(discounts, log_rhos, rewards, values,
                              bootstrap_value, clip_rho_threshold,
                              clip_pg_rho_threshold):
    """Calculates the ground truth for V-trace in Python/Numpy."""
    vs = []
    seq_len = len(discounts)
    rhos = np.exp(log_rhos)
    cs = np.minimum(rhos, 1.0)
    clipped_rhos = rhos
    if clip_rho_threshold:
        clipped_rhos = np.minimum(rhos, clip_rho_threshold)
    clipped_pg_rhos = rhos
    if clip_pg_rho_threshold:
        clipped_pg_rhos = np.minimum(rhos, clip_pg_rho_threshold)

    # This is a very inefficient way to calculate the V-trace ground truth.
    # We calculate it this way because it is close to the mathematical notation of
    # V-trace.
    # v_s = V(x_s)
    #       + \sum^{T-1}_{t=s} \gamma^{t-s}
    #         * \prod_{i=s}^{t-1} c_i
    #         * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
    # Note that when we take the product over c_i, we write `s:t` as the notation
    # of the paper is inclusive of the `t-1`, but Python is exclusive.
    # Also note that np.prod([]) == 1.
    values_t_plus_1 = np.concatenate([values, bootstrap_value[None, :]], axis=0)
    for s in range(seq_len):
        v_s = np.copy(values[s])  # Very important copy.
        for t in range(s, seq_len):
            v_s += (
                    np.prod(discounts[s:t], axis=0) * np.prod(cs[s:t],
                                                              axis=0) * clipped_rhos[t] *
                    (rewards[t] + discounts[t] * values_t_plus_1[t + 1] - values[t]))
        vs.append(v_s)
    vs = np.stack(vs, axis=0)
    pg_advantages = (
            clipped_pg_rhos * (rewards + discounts * np.concatenate(
        [vs[1:], bootstrap_value[None, :]], axis=0) - values))

    return impala_vtrace.VTraceReturns(vs=vs, pg_advantages=pg_advantages)


def _log_probs_from_logits_and_actions(batch_size):
    """Tests log_probs_from_logits_and_actions."""
    seq_len = 7
    num_actions = 3

    action_space = gym.spaces.Dict({'action1': gym.spaces.Discrete(num_actions)})

    policy_logits = convert_to_torch(_shaped_arange(seq_len, batch_size, num_actions) + 10, cast=None,
                                     device=None, in_place='try')
    actions = convert_to_torch(np.random.randint(
        0, num_actions, size=(seq_len, batch_size), dtype=np.int32), cast=None, device=None, in_place='try')

    distribution_mapper = DistributionMapper(action_space=action_space, distribution_mapper_config={})

    action_log_probs_tensor, _ = impala_vtrace.log_probs_from_logits_and_actions_and_spaces(
        policy_logits=[{'action1': policy_logits}], actions=[{'action1': actions}],
        action_spaces={0: gym.spaces.Dict({'action1': action_space})},
        distribution_mapper=distribution_mapper)
    action_log_probs_tensor = action_log_probs_tensor[0]['action1']
    # Ground Truth
    # Using broadcasting to create a mask that indexes action logits
    action_index_mask = np.array(actions[..., None]) == np.arange(num_actions)

    def index_with_mask(array, mask):
        return array[mask].reshape(*array.shape[:-1])

    # Note: Normally log(softmax) is not a good idea because it's not
    # numerically stable. However, in this test we have well-behaved values.
    ground_truth_v = index_with_mask(
        np.log(_softmax(np.array(policy_logits))), action_index_mask)

    assert np.allclose(ground_truth_v, action_log_probs_tensor)


def test_log_probs_from_logits_and_action():
    _log_probs_from_logits_and_actions(batch_size=1)
    _log_probs_from_logits_and_actions(batch_size=2)


def _vtrace(batch_size):
    """Tests V-trace against ground truth data calculated in python."""
    seq_len = 5

    # Create log_rhos such that rho will span from near-zero to above the
    # clipping thresholds. In particular, calculate log_rhos in [-2.5, 2.5),
    # so that rho is in approx [0.08, 12.2).
    log_rhos = _shaped_arange(seq_len, batch_size) / (batch_size * seq_len)
    log_rhos = 5 * (log_rhos - 0.5)  # [0.0, 1.0) -> [-2.5, 2.5).
    values = {
        'log_rhos': log_rhos,
        # T, B where B_i: [0.9 / (i+1)] * T
        'discounts':
            np.array([[0.9 / (b + 1)
                       for b in range(batch_size)]
                      for _ in range(seq_len)]),
        'rewards':
            _shaped_arange(seq_len, batch_size),
        'values':
            _shaped_arange(seq_len, batch_size) / batch_size,
        'bootstrap_value':
            _shaped_arange(batch_size) + 1.0,
        'clip_rho_threshold':
            3.7,
        'clip_pg_rho_threshold':
            2.2,
    }

    output = impala_vtrace.from_importance_weights(**values)

    ground_truth_v = _ground_truth_calculation(**values)
    for a, b in zip(ground_truth_v, output):
        assert np.allclose(a, b)


def test_vtrace():
    _vtrace(1)
    _vtrace(5)


def _vtrace_from_logits(batch_size):
    """Tests V-trace calculated from logits."""
    seq_len = 5
    num_actions = 3
    clip_rho_threshold = None  # No clipping.
    clip_pg_rho_threshold = None  # No clipping.

    # Intentionally leaving shapes unspecified to test if V-trace can
    # deal with that.

    values = {
        'behaviour_policy_logits':
            [{'action1': convert_to_torch(_shaped_arange(seq_len, batch_size, num_actions), device=None,
                                          cast=None, in_place='try')}],
        'target_policy_logits':
            [{'action1': convert_to_torch(_shaped_arange(seq_len, batch_size, num_actions), device=None,
                                          cast=None, in_place='try')}],
        'actions':
            [{'action1': convert_to_torch(np.random.randint(0, num_actions - 1, size=(seq_len, batch_size)),
                                          device=None, cast=None, in_place='try')}],
        'discounts':
            convert_to_torch(np.array(  # T, B where B_i: [0.9 / (i+1)] * T
                [[0.9 / (b + 1)
                  for b in range(batch_size)]
                 for _ in range(seq_len)]), device=None, cast=None, in_place='try'),
        'rewards':
            convert_to_torch(_shaped_arange(seq_len, batch_size), device=None, cast=None, in_place='try'),
        'values':
            [convert_to_torch(_shaped_arange(seq_len, batch_size) / batch_size, device=None, cast=None,
                              in_place='try')],
        'bootstrap_value':
            [convert_to_torch(_shaped_arange(batch_size) + 1.0, device=None, cast=None, in_place='try')],  # B
        'action_spaces': {0: gym.spaces.Dict({'action1': gym.spaces.Discrete(num_actions)})},
    }

    # initialize distribution mapper
    distribution_mapper = DistributionMapper(action_space=values["action_spaces"][0], distribution_mapper_config={})

    from_logits_output = impala_vtrace.from_logits(
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold, device=None,
        distribution_mapper=distribution_mapper,
        **values)

    target_log_probs, _ = impala_vtrace.log_probs_from_logits_and_actions_and_spaces(
        values['target_policy_logits'], values['actions'],
        action_spaces=values['action_spaces'], distribution_mapper=distribution_mapper)
    behaviour_log_probs, _ = impala_vtrace.log_probs_from_logits_and_actions_and_spaces(
        values['behaviour_policy_logits'], values['actions'],
        action_spaces=values['action_spaces'], distribution_mapper=distribution_mapper)
    log_rhos = impala_vtrace.get_log_rhos(target_log_probs, behaviour_log_probs)
    ground_truth_log_rhos, ground_truth_behaviour_action_log_probs, ground_truth_target_action_log_probs = \
        log_rhos, behaviour_log_probs, target_log_probs

    # Calculate V-trace using the ground truth logits.
    from_iw = impala_vtrace.from_importance_weights(
        log_rhos=ground_truth_log_rhos[0],
        discounts=values['discounts'],
        rewards=values['rewards'],
        values=values['values'][0],
        bootstrap_value=values['bootstrap_value'][0],
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold)

    assert np.allclose(from_iw.vs, from_logits_output.vs[0])
    assert np.allclose(from_iw.pg_advantages, from_logits_output.pg_advantages[0])
    assert np.allclose(ground_truth_behaviour_action_log_probs[0]['action1'],
                       from_logits_output.behaviour_action_log_probs[0]['action1'])
    assert np.allclose(ground_truth_target_action_log_probs[0]['action1'],
                       from_logits_output.target_action_log_probs[0]['action1'])
    assert np.allclose(ground_truth_log_rhos[0], from_logits_output.log_rhos[0])


def test_vtrace_from_logits():
    _vtrace_from_logits(1)
    _vtrace_from_logits(2)
