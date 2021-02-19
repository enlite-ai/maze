"""Implementation of the batching methods used for the impala algorithm. Specifically the method batch_output is given
    a list of ActorOutputs of the batch size. It is then tasked to batch these outputs together along the second
    dimension and convert everything to torch tensors. Finally it should return an ActorOutput (namedTuple) of the
    batched output"""
from collections import defaultdict
from typing import List, Union, Dict

import numpy as np
import torch

from maze.train.parallelization.distributed_actors.actor import ActorOutput


def batch_outputs_time_major(actor_outputs: List[ActorOutput], learner_device: str) -> ActorOutput:
    """Batch the collected output in time major format

    :param actor_outputs: A list of actor outputs (e.g. rollouts consisting of observations, actions_taken, infos,
        action_logtis, rewards and dones)
    :param learner_device: the device ('cpu' or 'cuda') of the learner

    :return: An ActorOutput Named tuple where the the list of input rollouts has been batched in the second dim.
    """
    new_output = {}
    for actor_output_field_name in ['observations', 'actions_taken', 'actions_logits']:
        step_keys = list(getattr(actor_outputs[0], actor_output_field_name).keys())
        new_stacked_field = {}
        for step_key in step_keys:
            collected = [getattr(actor_output, actor_output_field_name)[step_key] for actor_output in
                         actor_outputs]
            stacked = _batch_dict_lists(collected, learner_device)
            new_stacked_field[step_key] = stacked
        new_output[actor_output_field_name] = new_stacked_field

    for actor_output_field_name in ['rewards', 'dones']:
        collected = [getattr(actor_output, actor_output_field_name) for actor_output in actor_outputs]
        new_stacked_field = _batch_array_list(collected).to(learner_device)
        new_output[actor_output_field_name] = new_stacked_field

    for actor_output_field_name in ['infos']:
        collected = [getattr(actor_output, actor_output_field_name) for actor_output in actor_outputs]
        new_stacked_field = [[collected[batch_idx][time_idx] for batch_idx in range(len(collected))] for time_idx
                             in range(len(collected[0]))]
        new_output[actor_output_field_name] = new_stacked_field

    return ActorOutput(**new_output)


def _batch_array_list(array_list: List[Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
    """Batch together a list of arrays (either torch or numpy) after converting them to torch. That is ether stack them
        if the batch dimension does not exists, otherwise concatenate them in the batch dimension (2)

    :param array_list: A list of arrays (either torch or numpy)

    :return: the batched input
    """
    list_array = []
    for idx, array in enumerate(array_list):
        value = torch.from_numpy(array) if isinstance(array, np.ndarray) else array
        list_array.append(value)

    stacked_list = torch.cat(list_array, dim=1)
    return stacked_list


def _batch_dict_lists(dict_list: List[Dict[str, Union[np.ndarray, torch.Tensor]]], learner_device: str) \
        -> Dict[str, torch.Tensor]:
    """Batch together a list of dicts of arrays (either torch or numpy) after converting them to torch. That is ether
        stack them if the batch dimension does not exists, otherwise concatenate them in the batch dimension (2)

    :param dict_list: A list of dicts of arrays (either torch or numpy)
    :param learner_device: the device ('cpu' or 'gpu') of the learner

    :return: the batched input
    """
    list_dict = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            value = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            list_dict[k].append(value)

    stacked_dict = dict()
    for k in list_dict.keys():
        stacked_dict[k] = torch.stack(list_dict[k], dim=1).to(learner_device)
    return stacked_dict
