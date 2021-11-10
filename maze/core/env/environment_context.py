"""The environment context provides services that are globally available to all objects of the interaction loop.

The context is owned by the :obj:`~maze.core.core_env.CoreEnv`. Maintaining the global services in a separate object has
several benefits

- We avoid scattering references to the core env in all kinds of objects, which could easily result in unexpected
  behaviour.
- In scenarios with multiple instantiated agent-environment interaction loops the context can be used to identify
  the parent environments.
"""
import uuid
from typing import Callable


class EnvironmentContext:
    """
    This class keeps track of services that can be employed by all objects of the agent-environment loop.

    Currently the context is populated by

    - Event service: Acts as backend of the PubSub service, collects all events from the env.
      The event service can also be directly facilitated by components outside the environment (e.g. agent, heuristics,
      state/observation mapping)
    - Episode ID: Generates and keeps track of the current episode ID
      Episode IDs are used for connecting logged statistics, events and recorded trajectory data together, making
      analysis and drill-down across these different levels possible.
    - Step ID: Tracks ID of the core env step we are currently in. Helps wrappers recognize core env steps in multi-step
      scenarios.
    - Post-step callbacks: Handles registration of post-step callbacks and provides a utility to run them (usually
                           triggered from the outer-most wrapper)
    - Step lifecycle flags: Flags partially managed from the wrapper stack, helping the wrappers to determine
                            where we are in the step lifecycle and which one is the outer-most one
    """

    def __init__(self):
        from maze.core.events.event_service import EventService

        self.event_service = EventService()
        self.step_id = 0
        self._episode_id = None
        self._last_cleared_events_at = self.step_id

        # Step lifecycle flags - partially managed by wrappers to help them orient in the step lifecycle
        self.step_in_progress = False  # True during the whole env.step
        self.step_is_initiating = False  # True while descending through the wrapper stack, until the env time increment

        self._post_step_callbacks = []

    @property
    def episode_id(self) -> str:
        """
        Get the episode ID.

        Episode ID is a UUID generated in a lazy manner, ensuring that if the ID is not needed, the potentially costly
        random UUID generation is avoided. Once generated, it stays the same for the entire episode and then
        is reset.

        :return: Episode UUID as string
        """
        if self._episode_id is None:
            self._episode_id = str(uuid.uuid4())

        return self._episode_id

    def increment_env_step(self) -> None:
        """
        This must be called after the env step execution, to:
         - Clear event logs (used for statistics for the current step)
         - Increment step_id, which is used as env_time by default
        """
        self.step_id += 1
        self.step_is_initiating = False

    def reset_env_episode(self) -> None:
        """
        This must be called when resetting the environment, to notify the context about the start of a new episode.
        """
        self.step_id = 0
        self.step_in_progress = False
        self.step_is_initiating = False
        self._last_cleared_events_at = self.step_id

        self._episode_id = None  # Reset episode ID
        self.event_service.clear_events()
        self.event_service.clear_pubsub()

    def register_post_step(self, callback: Callable) -> None:
        """
        Register a function to be called after every single step, just before the events of the
        previous step are cleared.

        In case multiple env steps are initiated from a wrapper's step function (e.g. during step skipping),
        these callbacks will be called in between these individual steps as well.
        """
        self._post_step_callbacks.append(callback)

    def run_post_step_callbacks(self) -> None:
        """
        Run callbacks registered for post-step execution and then clear out the events from this step.
        """
        for callback in self._post_step_callbacks:
            callback()

        # Only clear events out if env time changed (important in multi-step scenarios)
        if self._last_cleared_events_at != self.step_id:
            self.event_service.clear_events()
            self._last_cleared_events_at = self.step_id

    def clone_from(self, context: 'EnvironmentContext') -> None:
        """ Clone environment by resetting to the provided context.

        :param context: The environment context to clone.
        """
        # cloning context.event_service is not required as this gets cleared prior to each step in
        # maze.core.env.environment_context.EnvironmentContext.pre_step
        # (as Python's GC cant resolve circular dependencies this would anyways cause severe problems)
        # self.event_service = copy.deepcopy(context.event_service)

        self.step_id = context.step_id
        self._episode_id = context._episode_id
