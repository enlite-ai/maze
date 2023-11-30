"""Broadcasting container for synchronizing policy updates across workers on local machine."""
from multiprocessing import RLock
from multiprocessing.managers import BaseManager
from typing import Dict, NoReturn

import cloudpickle


class BroadcastingContainer:
    """Synchronizes policy updates and other information across workers on local machine.

    Used for dummy and sub-process distribution scenarios.

    The BroadcastingContainer object can be read by all workers in order to update their policy, and it can be
    accessed by the main thread to write the updated policy from the learner into it.
    """

    def __init__(self):
        self._policy_version_counter = 0
        self._pickled_policy_state_dict = None
        self._stop_flag = False
        self._aux_data = None
        self._lock = RLock()

    def stop_flag(self) -> bool:
        """True if workers should exit."""
        with self._lock:
            return self._stop_flag

    def set_stop_flag(self) -> NoReturn:
        """Signal to the workers to exit after they finish the current rollout."""
        with self._lock:
            self._stop_flag = True

    def set_policy_state_dict(self, state_dict: Dict, aux_data: Dict = None) -> NoReturn:
        """Store new policy version.

        :param state_dict: New state dict to store
        :param aux_data: Dictionary with any auxiliary data to share
        """
        with self._lock:
            self._pickled_policy_state_dict = cloudpickle.dumps(state_dict)
            self._policy_version_counter += 1
            self._aux_data = aux_data

    def get_current_policy(self, last_version: int) -> (int, Dict, Dict):
        """Check if new policy version is available, and if so, return the state_dict and aux_data.

        :param last_version: Last version known to the worker/agent.
        :return: A tuple of (current_version_id, state_dict, aux_data). If the current version matches the last_version
                 supplied as the argument, the state_dict and aux_data will be nil, to avoid transferring
                 unnecessary data with each policy version check.
        """
        with self._lock:
            if self._policy_version_counter == last_version:
                return self._policy_version_counter, None, None

            return self._policy_version_counter, cloudpickle.loads(self._pickled_policy_state_dict), self._aux_data

    def policy_version(self) -> int:
        """Return the current policy version number (to check whether fetching a new state dict is necessary)."""
        return self._policy_version_counter


class BroadcastingManager(BaseManager):
    """A wrapper around BaseManager, used for managing the broadcasting container in multiprocessing scenarios."""
    pass
