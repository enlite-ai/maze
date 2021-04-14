"""An environment interface for multi-step, hierarchical and multi-agent environments."""
from abc import abstractmethod
from typing import Tuple, Union, Dict

from maze.core.env.base_env import BaseEnv

StepKeyType = Union[str, int]
ActorIDType = Tuple[StepKeyType, int]


class StructuredEnv(BaseEnv):
    """Interface for environments with sub-step structure, which is generally enough to cover multi-step,
    hierarchical and multi-agent environments.

    This environment can continuously create and destroy a previously unknown,
    unlimited number of actors during the course of an episode. Every actor is associated with one of the
    available policies.

    The lifecycle of the environment is decoupled from the lifecycle of the actors. The interaction loop should
    continue, until the environment as a whole is set to done, which is returned as usual by the step() function.
    Individual actors might end earlier, which can be queried by the is_actor_done() method.

    Pseudo-code of the interaction loop:

    # start a new episode
    observation = env.reset()

    while not done:
        # find out which actor is next to act (dictated by the env)
        sub_step_key, actor_id = env.actor_id()

        # obtain the next action from the policy
        action = sample_from_policy(observation, sub_step_key, actor_id)

        # step the env
        observation, reward, done, info = env.step(action)

        # optionally use is_actor_done() to find out if the actor was terminated (relevant during training)
    """

    @abstractmethod
    def actor_id(self) -> ActorIDType:
        """Returns the current sub step key along with the currently executed actor.

        The env must decide the actor in :meth:`~maze.core.env.base_env.BaseEnv.reset` and
        :meth:`~maze.core.env.base_env.BaseEnv.step`. In between these calls the return is
        constant per convention and :meth:`actor_id` can be called arbitrarily.

        Notes:
        * The id is unique only with respect to the sub step (every sub step may have its own actor 0).
        * Identities of done actors can not be reused in the same rollout.

        :return: The current actor, as tuple (sub step key, actor number).
        """

    @abstractmethod
    def is_actor_done(self) -> bool:
        """Returns True if the just stepped actor is done, which is different to the done flag of the environment.

        Like for :meth:`actor_id`, the env updates this flag in :meth:`~maze.core.env.base_env.BaseEnv.reset` and
        :meth:`~maze.core.env.base_env.BaseEnv.step`.

        :return: True if the actor is done.
        """

    @property
    @abstractmethod
    def agent_counts_dict(self) -> Dict[StepKeyType, int]:
        """Returns the maximum count of agents per sub-step that the environment features.

        If the agent count for a particular sub-step is dynamic (unknown upfront), then returns -1 for this sub-step.
        This then limits available training configurations (e.g. with a dynamic number of agents, only
        a shared policy can be trained -- not multiple separate per-agent policies, as their count is not
        known upfront).

        For example:
          - For a vehicle-routing environment where max 5 driver agents will get to act during sub-step 0,
            this method should return {0: 5}
          - For a vehicle routing environment where a dynamic number of agents will get to act during sub-step 0,
            this method should return {0: -1}
          - For a two-step cutting environment where a piece is selected during sub-step 0 and then cut during
            sub-step 1 (with just one selection and cut happening in each step),
            this method should return {0: 1, 1: 1}
        """
