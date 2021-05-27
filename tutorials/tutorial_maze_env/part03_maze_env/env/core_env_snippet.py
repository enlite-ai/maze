from maze.core.env.structured_env import ActorID


class Cutting2DCoreEnvironment(CoreEnv):

    ...

    def is_actor_done(self) -> bool:
        """Returns True if the just stepped actor is done, which is different to the done flag of the environment."""
        return False

    def actor_id(self) -> ActorID:
        """Returns the currently executed actor along with the policy id. The id is unique only with
        respect to the policies (every policy has its own actor 0).
        Note that identities of done actors can not be reused in the same rollout.

        :return: The current actor, as tuple (policy id, actor number).
        """
        return ActorID(step_key=0, agent_id=0)

    ...