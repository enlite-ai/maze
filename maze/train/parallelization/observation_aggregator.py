from abc import ABC, abstractmethod
from typing import List, Any, Dict

import numpy as np

from maze.train.utils.train_utils import stack_numpy_dict_list


class ObservationAggregator(ABC):
    """Observation aggregator used in distributed training
    for aggregating observations of multiple instances of the same environment.
    """

    def __init__(self):
        self.observations = []

    def reset(self, observations: List[Any] = None) -> None:
        """Reset aggregator.

        :param observations: a list of observations.
        """
        if observations is None:
            self.observations = []
        else:
            self.observations = observations

    @abstractmethod
    def aggregate(self) -> Any:
        """This function aggregates the collected list of observations.
        """
        pass


class DictObservationAggregator(ObservationAggregator):
    """Dictionary observation aggregator.
    """

    def aggregate(self) -> Dict[str, np.array]:
        """Stack list of dictionary observations per key along axis=0.
        """
        return stack_numpy_dict_list(self.observations, expand=True)
