from dataclasses import dataclass
from typing import Dict, Optional, TypeVar, Tuple, List, Union

import numpy as np

from maze.core.log_events.step_event_log import StepEventLog
from maze.core.log_stats.log_stats import LogStats
from maze.perception.perception_utils import convert_to_numpy, convert_to_torch
from maze.train.utils.train_utils import stack_numpy_dict_list

StepKeyType = TypeVar('StepKeyType', str, int)


@dataclass
class SpacesStepRecord:
    observations: Dict[StepKeyType, Dict[str, np.ndarray]]
    actions: Dict[StepKeyType, Dict[str, np.ndarray]]
    rewards: Optional[Dict[StepKeyType, Union[float, np.ndarray]]]
    dones: Optional[Dict[StepKeyType, Union[float, bool]]]
    infos: Optional[Dict[StepKeyType, Dict]] = None

    logits: Optional[Dict[StepKeyType, Dict[str, np.ndarray]]] = None
    event_log: Optional[StepEventLog] = None
    stats: Optional[LogStats] = None

    batch_shape: Optional[List[int]] = None

    def is_batched(self):
        return self.batch_shape is not None

    def to_numpy(self):
        self.observations = convert_to_numpy(self.observations, cast=None, in_place=True)
        self.actions = convert_to_numpy(self.actions, cast=None, in_place=True)
        self.rewards = convert_to_numpy(self.rewards, cast=None, in_place=True)
        self.dones = convert_to_numpy(self.dones, cast=None, in_place=True)

        if self.logits is not None:
            self.logits = convert_to_numpy(self.logits, cast=None, in_place=True)

        return self

    def to_torch(self, device: str):
        self.observations = convert_to_torch(self.observations, device=device, cast=None, in_place=True)
        self.actions = convert_to_torch(self.actions, device=device, cast=None, in_place=True)
        self.rewards = convert_to_torch(self.rewards, device=device, cast=None, in_place=True)
        self.dones = convert_to_torch(self.dones, device=device, cast=None, in_place=True)

        if self.logits is not None:
            self.logits = convert_to_torch(self.logits, device=device, cast=None, in_place=True)

        return self

    @classmethod
    def stack_records(cls, records: List['SpacesStepRecord']):
        logits_present = records[0].logits is not None

        stacked_record = SpacesStepRecord(
            observations={}, actions={}, logits={} if logits_present else None,
            rewards=stack_numpy_dict_list([r.rewards for r in records]),
            dones=stack_numpy_dict_list([r.dones for r in records]))

        # Actions and observations are nested dict spaces => need to go one level down with stacking
        for step_key in records[0].observations.keys():
            stacked_record.actions[step_key] = stack_numpy_dict_list([r.actions[step_key] for r in records],
                                                                     expand=True)
            stacked_record.observations[step_key] = stack_numpy_dict_list([r.observations[step_key] for r in records],
                                                                          expand=True)

            if logits_present:
                stacked_record.logits[step_key] = stack_numpy_dict_list(
                    [r.logits[step_key] for r in records],
                    expand=True)

        stacked_record.batch_shape = [len(records)] + records[0].batch_shape if records[0].batch_shape \
            else [len(records)]

        return stacked_record
