"""An environment interface for multi-step, hierarchical and multi-agent environments."""
from abc import abstractmethod
from typing import Tuple, Union

from maze.core.env.base_env import BaseEnv


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
    def actor_id(self) -> Tuple[Union[str, int], int]:
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
