from dataclasses import dataclass
from typing import Dict, Optional, TypeVar, Tuple, List

import numpy as np

from maze.core.log_events.step_event_log import StepEventLog
from maze.core.log_stats.log_stats import LogStats
from maze.train.utils.train_utils import stack_numpy_dict_list

StepKeyType = TypeVar('StepKeyType', str, int)


@dataclass
class SpacesStepRecord:
    observations: Dict[StepKeyType, np.ndarray]
    actions: Dict[StepKeyType, np.ndarray]
    rewards: Optional[Dict[StepKeyType, float]]
    dones: Optional[Dict[StepKeyType, bool]]
    infos: Optional[Dict[StepKeyType, Dict]] = None
    event_log: Optional[StepEventLog] = None
    stats: Optional[LogStats] = None

    # logits: Optional[Dict[StepKeyType, np.ndarray]]
    batch_shape: Optional[List[int]] = None

    def is_batched(self):
        return self.batch_shape is not None

    @classmethod
    def stack_records(cls, records: List['SpacesStepRecord']):
        stacked_record = SpacesStepRecord(
            observations={}, actions={},
            rewards=stack_numpy_dict_list([r.rewards for r in records]),
            dones=stack_numpy_dict_list([r.dones for r in records]))

        # Actions and observations are nested dict spaces => need to go one level down with stacking
        for step_key in records[0].observations.keys():
            stacked_record.actions[step_key] = stack_numpy_dict_list([r.actions[step_key] for r in records],
                                                                     expand=True)
            stacked_record.observations[step_key] = stack_numpy_dict_list([r.observations[step_key] for r in records],
                                                                          expand=True)

        stacked_record.batch_shape = [len(records)] + records[0].batch_shape if records[0].batch_shape \
            else [len(records)]

        return stacked_record
