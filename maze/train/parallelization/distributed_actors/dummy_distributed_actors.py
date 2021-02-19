"""Dummy distributed actors. Ran sequentially in the main process."""

import time
from typing import Callable, Union, List, Tuple, Dict

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.perception.perception_utils import convert_to_torch
from maze.train.parallelization.distributed_actors.actor import ActorAgent, ActorOutput
from maze.train.parallelization.distributed_actors.broadcasting_container import BroadcastingContainer
from maze.train.parallelization.distributed_actors.distributed_actors import BaseDistributedActors
from maze.train.trainers.impala.impala_batching import batch_outputs_time_major
from maze.utils.bcolors import BColors


class DummyDistributedActors(BaseDistributedActors):
    """Dummy implementation of distributed actors creates the actors as a list. Once the outputs are to
        be collected, it simply rolls them out in a loop until is has enough to be returned."""

    def __init__(self,
                 env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
                 policy: TorchPolicy,
                 n_rollout_steps: int,
                 n_actors: int,
                 batch_size: int):
        super().__init__(env_factory, policy, n_rollout_steps, n_actors, batch_size)

        self.broadcasting_container = BroadcastingContainer()
        self.current_actor_idx = 0

        self.actors: List[ActorAgent] = []
        self.policy_version_counters = []

        for ii in range(self.n_actors):
            actor = ActorAgent(env_factory, policy, n_rollout_steps)
            self.actors.append(actor)
            self.policy_version_counters.append(0)

        if self.n_actors > self.batch_size:
            BColors.print_colored(
                f'It does not make much sense to have more actors (given value: {n_actors}) than '
                f'the actor_batch_size (given value: {batch_size}) when using the DummyMultiprocessingModule.',
                color=BColors.WARNING)

    @override(BaseDistributedActors)
    def start(self) -> None:
        """Nothing to do in dummy implementation"""
        pass

    @override(BaseDistributedActors)
    def stop(self) -> None:
        """Nothing to do in dummy implementation"""
        pass

    @override(BaseDistributedActors)
    def broadcast_updated_policy(self, state_dict: Dict) -> None:
        """Store the newest policy in the shared network object"""
        converted_state_dict = convert_to_torch(state_dict, in_place=False, cast=None, device=self.policy.device)
        self.broadcasting_container.set_policy_state_dict(converted_state_dict)

    @override(BaseDistributedActors)
    def collect_outputs(self, learner_device: str) -> Tuple[ActorOutput, float, float, float]:
        """Run the rollouts and collect the outputs."""

        actor_outputs = []

        start_wait_time = time.time()

        while len(actor_outputs) < self.batch_size:

            # Update the policy of the actor if a new version of the policy has been published by the learner
            shared_policy_version_counter = self.broadcasting_container.policy_version()
            if self.policy_version_counters[self.current_actor_idx] < shared_policy_version_counter:
                self.actors[self.current_actor_idx].update_policy(self.broadcasting_container.policy_state_dict())
                self.policy_version_counters[self.current_actor_idx] = shared_policy_version_counter

            output = self.actors[self.current_actor_idx].rollout()
            actor_outputs.append(ActorOutput(*output[:-1]))
            self.current_actor_idx = self.current_actor_idx + 1 if self.current_actor_idx < len(self.actors) - 1 else 0

            # collect episode statistics
            for stat in output.stats:
                if stat is not None:
                    self.epoch_stats.receive(stat)

        dequeue_time = time.time() - start_wait_time

        return batch_outputs_time_major(actor_outputs=actor_outputs, learner_device=learner_device), 0, 0, dequeue_time
