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

from maze.utils.bcolors import BColors


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
    """

    def __init__(self):
        from maze.core.events.event_service import EventService

        self.event_service = EventService()
        self.step_id = 0
        self._episode_id = None

        self._should_clear_events = True

        self._pre_step_callbacks = []
        self._increment_env_step_warning_printed = False

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
        if self._should_clear_events and not self._increment_env_step_warning_printed:
            BColors.print_colored("Events have not been cleared at the start of the current step!"
                                  "If you called the step function inside a wrapper, look into the "
                                  "`Wrapper.keep_inner_hooks` flag.", BColors.WARNING)
            # log only once
            self._increment_env_step_warning_printed = True

        self.step_id += 1
        self._should_clear_events = True

    def reset_env_episode(self) -> None:
        """
        This must be called when resetting the environment, to notify the context about the start of a new episode.
        """
        self.step_id = 0
        self._episode_id = None  # Reset episode ID
        self.event_service.clear_events()
        self.event_service.clear_pubsub()

    def register_pre_step(self, callback: Callable) -> None:
        """
        Register a function to be called before every single step, just before the events of the
        previous step are cleared.
        """
        self._pre_step_callbacks.append(callback)

    def pre_step(self) -> None:
        """Prepare the event system for a new step.

        Checks internally if this has already been done for the current env step, in this case nothing happens.
        """
        if not self._should_clear_events:
            return

        for callback in self._pre_step_callbacks:
            callback()

        self._should_clear_events = False
        self.event_service.clear_events()

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

        self._should_clear_events = context._should_clear_events

        self._increment_env_step_warning_printed = context._increment_env_step_warning_printed
