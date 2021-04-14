"""Executes the provided policies in an Agent Integration setting."""
import traceback
from collections import namedtuple
from threading import Event

from maze.core.agent.policy import Policy
from maze.core.agent_integration.external_core_env import ExternalCoreEnv
from maze.core.agent_integration.maze_action_candidates import ActionCandidates
from maze.core.log_stats.log_stats_env import LogStatsEnv

ExceptionReport = namedtuple("ExceptionReport", "exception traceback")
"""Tuple for passing error back to the main thread."""


class AgentExecution:
    """Executes the provided policies in an Agent Integration setting.

    Policies are executed until the rollout_done event is set, indicating that the rollout has been finished.
    Then, a final reset is sent and execution stops. Expected to be run on a separate thread alongside the
    agent integration running on the main thread.

    :param env: Environment to step.
    :param policy: Structured policy working with structured environments.
    :param rollout_done_event: event indicating that the rollout has been finished.
    """

    def __init__(self,
                 env: ExternalCoreEnv,
                 policy: Policy,
                 rollout_done_event: Event,
                 num_candidates: int):
        self.env = env
        self.policy = policy
        self.rollout_done_event = rollout_done_event
        self.num_candidates = num_candidates

    def run_rollout_maze(self):
        """Step the environment until the rollout is done."""
        try:
            # We need to reset first, otherwise no observation is available
            observation = self.env.reset()
            while not self.rollout_done_event.is_set():
                actor_id = self.env.actor_id()

                maze_state = self.env.get_maze_state() if self.policy.needs_state() else None
                env = self.env if self.policy.needs_env() else None

                # Get either a single action or multiple candidates wrapped in action candidates object
                if self.num_candidates > 1:
                    action = ActionCandidates(self.policy.compute_top_action_candidates(
                        observation=observation,
                        maze_state=maze_state,
                        env=env,
                        num_candidates=self.num_candidates,
                        actor_id=actor_id)
                    )
                else:
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
            self.env.maze_action_queue.put(exception_report)
            raise
