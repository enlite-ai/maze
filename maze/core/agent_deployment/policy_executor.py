"""Executes the provided policies in an Agent Deployment setting."""
import traceback
from collections import namedtuple
from queue import Queue
from threading import Event

from maze.core.agent.policy import Policy
from maze.core.agent_deployment.external_core_env import ExternalCoreEnv
from maze.core.log_stats.log_stats_env import LogStatsEnv

ExceptionReport = namedtuple("ExceptionReport", "exception traceback")
"""Tuple for passing error back to the main thread."""


class PolicyExecutor:
    """Executes the provided policies in an Agent Deployment setting.

    Policies are executed until the rollout_done event is set, indicating that the rollout has been finished.
    Then, a final reset is sent and execution stops. Expected to be run on a separate thread alongside the
    agent deployment running on the main thread.

    :param env: Environment to step.
    :param policy: Structured policy working with structured environments.
    :param rollout_done_event: event indicating that the rollout has been finished.
    """

    def __init__(self,
                 env: ExternalCoreEnv,
                 policy: Policy,
                 rollout_done_event: Event,
                 exception_queue: Queue):
        self.env = env
        self.policy = policy
        self.rollout_done_event = rollout_done_event
        self.exception_queue = exception_queue

    def run_rollout_loop(self):
        """Step the environment until the rollout is done."""
        try:
            # We need to reset first, otherwise no observation is available
            observation = self.env.reset()
            while not self.rollout_done_event.is_set():
                actor_id = self.env.actor_id()

                maze_state = self.env.get_maze_state() if self.policy.needs_state() else None
                env = self.env if self.policy.needs_env() else None

                # Get action from the policy
                action = self.policy.compute_action(
                    observation=observation,
                    maze_state=maze_state,
                    env=env,
                    actor_id=actor_id,
                    deterministic=True)

                observation, _, done, _ = self.env.step(action)
            # Final reset required to notify all wrappers.
            self.env.reset()

            # Notify stats logging about epoch end if available
            if isinstance(self.env, LogStatsEnv):
                self.env.write_epoch_stats()

        except Exception as exception:
            # Send exception along with a traceback to the main thread
            exception_report = ExceptionReport(exception, traceback.format_exc())
            self.exception_queue.put(exception_report)
            raise
