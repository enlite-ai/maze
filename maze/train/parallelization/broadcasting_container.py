"""Broadcasting container for synchronizing policy updates across workers on local machine."""
from multiprocessing.managers import BaseManager
from typing import Dict, NoReturn, Optional

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

    def stop_flag(self) -> bool:
        """True if workers should exit."""
        return self._stop_flag

    def set_stop_flag(self) -> NoReturn:
        """Signal to the workers to exit after they finish the current rollout."""
        self._stop_flag = True

    def policy_version(self) -> int:
        """Return the current policy version number (to check whether fetching a new state dict is necessary)."""
        return self._policy_version_counter

    def policy_state_dict(self) -> Dict:
        """Return the current policy state dict."""
        return cloudpickle.loads(self._pickled_policy_state_dict)

    def set_policy_state_dict(self, state_dict: Dict, aux_data: Dict = None) -> NoReturn:
        """Store new policy version.

        :param state_dict: New state dict to store
        :param aux_data: Dictionary with any auxiliary data to share
        """
        self._pickled_policy_state_dict = cloudpickle.dumps(state_dict)
        self._policy_version_counter += 1
        self._aux_data = aux_data

    def aux_data(self) -> Optional[Dict]:
        """Return auxiliary data stored together with the policy."""
        return self._aux_data


class BroadcastingManager(BaseManager):
    """A wrapper around BaseManager, used for managing the broadcasting container in multiprocessing scenarios."""
    pass
