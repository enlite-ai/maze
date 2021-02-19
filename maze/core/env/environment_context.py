"""The environment context provides services that are globally available to all objects of the interaction loop.

The context is owned by the :obj:`~maze.core.core_env.CoreEnv`. Maintaining the global services in a separate object has
several benefits

- We avoid scattering references to the core env in all kinds of objects, which could easily result in unexpected
  behaviour.
- In scenarios with multiple instantiated agent-environment interaction loops the context can be used to identify
  the parent environments.
"""

import uuid


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
        This must be called after the env step execution, to notify the services about the start of a new step.
        """
        self.step_id += 1
        self.event_service.notify_next_step()

    def reset_env_episode(self):
        """
        This must be called when resetting the environment, to notify the context about the start of a new episode.
        """
        self.step_id = 0
        self._episode_id = None  # Reset episode ID
