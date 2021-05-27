"""
The aim of the agent integration is to allow for a seamless integration (execution) of policies
and the required pre-processing and post-processing stack in production systems. A second use case is
the training on environments that cannot be stepped by the interaction loop and require inversion
of control (e.g. Unity Engine).

When using the agent integration, the control flow is inverted. Normally, the environment is stepped by the agent.
With agent integration, the agent is queried for MazeActions instead, i.e. it is the (external) environment that controls
the flow, not the agent.

This is done while reusing most of the standard architecture used in the normal control flow, like wrappers.
Agent, the wrapper stack and flat env run on a separate thread together with a special core env that obtains
states from the agent integration and passes MazeActions back.
"""
from queue import Queue
from threading import Event, Thread
from typing import Any, Dict, Union, Tuple, List, Type, Optional

import numpy as np

from maze.core.agent.policy import Policy
from maze.core.agent_integration.agent_execution import AgentExecution, ExceptionReport
from maze.core.agent_integration.maze_action_candidates import ActionConversionCandidatesInterface
from maze.core.agent_integration.external_core_env import ExternalCoreEnv
from maze.core.env.action_conversion import ActionConversionInterface
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationConversionInterface
from maze.core.env.structured_env import ActorID
from maze.core.events.event_record import EventRecord
from maze.core.rendering.renderer import Renderer
from maze.core.wrappers.wrapper import Wrapper


class AgentIntegration:
    """Encapsulates an agent, space interfaces and a stack of wrappers, to make the agent's MazeActions accessible to
    an external env.

    External env should supply states to agent integration object, and can query it for agent MazeActions. The
    agent with the supplied policy (or multiple policies) is run on a separate thread.

    Note that the two threads (main thread running this wrapper and the second thread running the agent, wrappers etc.)
    never run in parallel, i.e. one is always suspended. This is enforced using the queues. Either the main thread
    runs and the agent thread is waiting for the state to be passed from the main thread, or the agent thread is
    running (computing the MazeAction) and the main thread is waiting until the MazeAction is passed back (then,
    the second thread is suspended again until the next state is passed in via the queue).

    Queues have max size of one, enforcing that one step can be taken at a time.

    :param policy: Structured policy working with structured environments.
                   When querying for MazeAction, it can be specified what policy should be run
                   (using the actor_id parameter, first part of which corresponds to the policy_id).
    :param action_conversions: Action conversion interfaces for the respective policies.
    :param observation_conversions: Observation interfaces for the respective policies.
    :param num_candidates: Number of MazeAction candidates to get from the policy. If greater than 1, will return
                           multiple MazeActions wrapped in
                           :class:`~maze.core.agent_integration.maze_action_candidates.MazeActionCandidates`
    :param wrapper_types: Which wrappers should be run as part of the agent's stack.
    :param wrapper_kwargs: Optional arguments to pass to the given wrappers on instantiation.
    """

    def __init__(self,
                 policy: Policy,
                 action_conversions: Dict[Union[str, int], ActionConversionInterface],
                 observation_conversions: Dict[Union[str, int], ObservationConversionInterface],
                 num_candidates: int = 1,
                 wrapper_types: Optional[List[Type[Wrapper]]] = None,
                 wrapper_kwargs: Optional[List[Dict[str, Any]]] = None,
                 renderer: Optional[Renderer] = None):
        self.rollout_done = False

        # Thread synchronisation
        self.state_queue = Queue(maxsize=1)
        self.maze_action_queue = Queue(maxsize=1)
        self.rollout_done_event = Event()

        # If we are working with multiple candidate actions, wrap the action_conversion interface
        if num_candidates > 1:
            for policy_id, action_conversion in action_conversions.items():
                action_conversions[policy_id] = ActionConversionCandidatesInterface(action_conversion)

        # Env & wrappers
        self.external_env = ExternalCoreEnv(self.state_queue, self.maze_action_queue, self.rollout_done_event,
                                            renderer=renderer)
        self.wrapped_env = MazeEnv(self.external_env, action_conversions, observation_conversions)
        if wrapper_types is None:
            wrapper_types = []
        assert wrapper_kwargs is None or len(wrapper_kwargs) == len(wrapper_types), \
            "Count of provided wrapper key-word argument dicts must correspond to the count of provided wrappers."
        for i, wrapper_type in enumerate(wrapper_types):
            kwargs = wrapper_kwargs[i] if wrapper_kwargs is not None else {}
            self.wrapped_env = wrapper_type.wrap(self.wrapped_env, **kwargs)

        # Agent/policy wrapper
        self.agent = AgentExecution(self.wrapped_env, policy, self.rollout_done_event, num_candidates)
        self.agent_thread = Thread(target=self.agent.run_rollout_maze, daemon=True)
        self.agent_thread.start()

    def get_maze_action(self,
                        maze_state: MazeStateType,
                        reward: Union[None, float, np.ndarray, Any],
                        done: bool,
                        info: Union[None, Dict[Any, Any]],
                        events: Optional[List[EventRecord]] = None,
                        actor_id: ActorID = ActorID(0, 0)):
        """Query the agent for MazeAction derived from the given state.

        Passes the state etc. to the agent's thread, where it is integrated into an ordinary env rollout loop.
        In the first step, an env reset call is propagated through the env wrapper stack on agent's thread.

        :param maze_state: Current state of the environment.
        :param reward: Reward for the previous step (can be null in initial step)
        :param done: Whether the external environment is done
        :param info: Info dictionary
        :param events: List of events to be recorded for this step (mainly useful for statistics and event logs)
        :param actor_id: Optional ID of the actor to run next (comprised of policy_id and agent_id)
        :return: MazeAction from the agent
        """
        if self.rollout_done:
            raise RuntimeError("External env has been declared done already. Please create a new connector object for"
                               "a new episode.")

        self.external_env.set_actor_id(actor_id)
        self.state_queue.put((maze_state, reward, done, info, events))
        # Here, the MazeAction is suspended until the agent on the second thread runs another step and the MazeAction
        # is passed back through the MazeAction queue.
        maze_action = self.maze_action_queue.get()

        # If exception occurs in the agent thread, it will be passed back using this same queue as an exception report.
        if isinstance(maze_action, ExceptionReport):
            exc_report = maze_action
            raise RuntimeError("Error encountered in agent thread:\n" + exc_report.traceback) from exc_report.exception

        return maze_action

    def finish_rollout(self,
                       maze_state: MazeStateType,
                       reward: Union[float, np.ndarray, Any],
                       done: bool,
                       info: Dict[Any, Any],
                       events: Optional[List[EventRecord]] = None):
        """
        Should be called when the rollout is finished. While this has no effect on the provided MazeActions,
        it passes an env reset call through the wrapper stack, enabling the wrappers to do any work they
        normally do at the end of an episode (like write trajectory data).

        :param maze_state: Final state of the rollout
        :param reward: Reward for the previous step (can be null in initial step)
        :param done: Whether the external environment is done
        :param info: Info dictionary
        :param events: List of events to be recorded for this step (mainly useful for statistics and event logs)
        """
        self.rollout_done = True
        self.rollout_done_event.set()
        self.state_queue.put((maze_state, reward, done, info, events))
        self.agent_thread.join()
