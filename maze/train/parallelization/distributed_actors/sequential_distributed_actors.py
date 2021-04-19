"""Dummy distributed actors. Ran sequentially in the main process."""

import time
from typing import Callable, Union, List, Tuple, Dict

from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.rollout.rollout_generator import RolloutGenerator
from maze.core.trajectory_recording.records.structured_spaces_record import StructuredSpacesRecord
from maze.core.trajectory_recording.records.trajectory_record import SpacesTrajectoryRecord
from maze.perception.perception_utils import convert_to_torch
from maze.train.parallelization.distributed_actors.broadcasting_container import BroadcastingContainer
from maze.train.parallelization.distributed_actors.distributed_actors import DistributedActors
from maze.utils.bcolors import BColors


class SequentialDistributedActors(DistributedActors):
    """Dummy implementation of distributed actors creates the actors as a list. Once the outputs are to
        be collected, it simply rolls them out in a loop until is has enough to be returned.

    :param actor_env_seeds: A list of seeds for each actors' env.
    """

    def __init__(self,
                 env_factory: Callable[[], Union[StructuredEnv, StructuredEnvSpacesMixin, LogStatsEnv]],
                 policy: TorchPolicy,
                 n_rollout_steps: int,
                 n_actors: int,
                 batch_size: int,
                 actor_env_seeds: List[int]):
        super().__init__(env_factory, policy, n_rollout_steps, n_actors, batch_size)

        self.broadcasting_container = BroadcastingContainer()
        self.current_actor_idx = 0

        self.actors: List[RolloutGenerator] = []
        self.policy_version_counter = 0

        for env_seed in actor_env_seeds:
            env = env_factory()
            env.seed(env_seed)
            actor = RolloutGenerator(env=env, record_logits=True, record_episode_stats=True)
            self.actors.append(actor)

        if self.n_actors > self.batch_size:
            BColors.print_colored(
                f'It does not make much sense to have more actors (given value: {n_actors}) than '
                f'the actor_batch_size (given value: {batch_size}) when using the DummyMultiprocessingModule.',
                color=BColors.WARNING)

    @override(DistributedActors)
    def start(self) -> None:
        """Nothing to do in dummy implementation"""
        pass

    @override(DistributedActors)
    def stop(self) -> None:
        """Nothing to do in dummy implementation"""
        pass

    @override(DistributedActors)
    def broadcast_updated_policy(self, state_dict: Dict) -> None:
        """Store the newest policy in the shared network object"""
        converted_state_dict = convert_to_torch(state_dict, in_place=False, cast=None, device=self.policy.device)
        self.policy.load_state_dict(converted_state_dict)

    @override(DistributedActors)
    def collect_outputs(self, learner_device: str) -> Tuple[StructuredSpacesRecord, float, float, float]:
        """Run the rollouts and collect the outputs."""
        start_wait_time = time.time()
        trajectories = []

        while len(trajectories) < self.batch_size:
            trajectory = self.actors[self.current_actor_idx].rollout(policy=self.policy, n_steps=self.n_rollout_steps)

            # collect episode statistics
            for record in trajectory.step_records:
                if record.episode_stats is not None:
                    self.epoch_stats.receive(record.episode_stats)

            trajectories.append(trajectory)
            self.current_actor_idx = self.current_actor_idx + 1 if self.current_actor_idx < len(self.actors) - 1 else 0

        stacked_record = SpacesTrajectoryRecord.stack_trajectories(trajectories).stack()
        dequeue_time = time.time() - start_wait_time
        return stacked_record.to_torch(device=learner_device), 0, 0, dequeue_time
